'''
 File name: cross_validation.py
 Author: ML Wizards
 Date created: 29/10/2023
 Date last modified: 29/10/2023
 Python Version: 3.9.18
 '''

import numpy as np
from implementations import *


def hyper_param_tuning(y_train, x_train, k_fold, gammas, lambdas, seed=1):
    """cross validation over regularisation parameter lambda and learning rate lambda.

    Args:
        y_train: shape=(N,)
        x_train: shape=(N, D)
        k_fold: integer, the number of folds
        gammas: shape = (q, ) where q is the number of values of gamma to test
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        seed: integer, random seed
    Returns:
        best_gamma : scalar, value of the best gamma
        best_lambda : scalar, value of the best lambda
        best_loss : scalar
    """

    # split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)

    # cross validation over gammas and lambdas: 
    best_lambdas = []
    best_f1s = []

    # vary gammas
    for gamma in gammas:
        # cross validation
        f1s = []
        for lambda_ in lambdas:
            f1s_tmp = []
            for k in range(k_fold):
                loss, f1 = hyper_param_cross_validation(y_train, x_train, k_indices, k, lambda_, gamma)
                f1s_tmp.append(f1)
            f1s.append(np.mean(f1s_tmp))

        ind_lambda_opt = np.argmax(f1s)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_f1s.append(f1s[ind_lambda_opt])

    # find best gamma and lambda
    ind_best_gamma = np.argmax(best_f1s)
    best_gamma = gammas[ind_best_gamma]
    best_lambda = best_lambdas[ind_best_gamma]
    best_f1 = best_f1s[ind_best_gamma]

    return best_gamma, best_lambda, best_f1


def models_cross_validation(y_train, x_train, k_fold, seed=1):
    """Returns the mean and the std of the accuracies and f1-scores for each model.

    Args:
        y_train:    shape=(N,)
        x_train:    shape=(N,)
        k_fold:     scalar
        seed:       scalar
    
    Returns:
        accuracies_mean:    a list containing the mean of the accuracies for each model
        accuracies_std:     a list containing the std of the accuracies for each model
        f1s_mean:           a list containing the mean of the f1-scores for each model
        f1s_std:            a list containing the std of the f1-scores for each model
    """
    # split data in k fold
    k_indices = build_k_indices(y_train, k_fold, seed)

    # accuracies for each model
    accuracies_random = []
    accuracies_lg = []
    accuracies_rlg = []
    accuracies_wrlg = []
    
    # f1-scores for each model
    f1s_random = []
    f1s_lg = []
    f1s_rlg = []
    f1s_wrlg = []

    # vary folds
    for k in range(k_fold):
        accuracy, f1 = predictions(y_train, x_train, k_indices, k)
        accuracies_random.append(accuracy[0])
        accuracies_lg.append(accuracy[1])
        accuracies_rlg.append(accuracy[2])
        accuracies_wrlg.append(accuracy[3])
        f1s_random.append(f1[0])
        f1s_lg.append(f1[1])
        f1s_rlg.append(f1[2])
        f1s_wrlg.append(f1[3])
    
    # Calculate the mean and the std of the accuracies for each model
    accuracies_mean = [np.mean(accuracies_random), np.mean(accuracies_lg), np.mean(accuracies_rlg), np.mean(accuracies_wrlg)]
    accuracies_std = [np.std(accuracies_random),  np.std(accuracies_lg), np.std(accuracies_rlg), np.std(accuracies_wrlg)]
    
    # Calculate the mean and the std of the f1-scores for each model
    f1s_mean = [np.mean(f1s_random), np.mean(f1s_lg), np.mean(f1s_rlg), np.mean(f1s_wrlg)]
    f1s_std = [np.std(f1s_random), np.std(f1s_lg), np.std(f1s_rlg), np.std(f1s_wrlg)]

    return accuracies_mean, accuracies_std, f1s_mean, f1s_std


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def f1_score(y_te, y_pred):
    """Computes the F1 score.

    Args:
        y_te:    the true labels, shape=(N,)
        y_pred:  the predicted labels, shape=(N,)

    Returns:
        a scalar value of the F1 score.
    """
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = sum(1 for a, b in zip(y_te, y_pred) if a == 1 and b == 1)
    FP = sum(1 for a, b in zip(y_te, y_pred) if (a == 0 or a == -1) and b == 1)
    FN = sum(1 for a, b in zip(y_te, y_pred) if a == 1 and (b == 0 or b == -1))
    

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def hyper_param_cross_validation(y, x, k_indices, k, lambda_, gamma):
    """ returns the f1 score for a given k'th subgroup in test, others in train

    Args:
        y:          shape=(N,)
        x:          shape=(N, D)
        k_indices:  shape=(K, N/K)
        k:          the k'th subgroup in test
        lambda_:    the regularisation parameter
        gamma:      the learning rate

    Returns:
        loss: scalar
        f1: scalar
    """

    # get k'th subgroup in test, others in train: 
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k])

    
    # weighted regularized logistic regression: 
    tx = np.c_[np.ones(y_train.shape[0]), x_train]
    initial_w = np.zeros(tx.shape[1])
    w, loss = weighted_reg_logistic_regression(y_train, tx, lambda_, initial_w, 301, gamma)
    
    # calculate the f1 score: 
    tx_test = np.c_[np.ones(y_test.shape[0]), x_test]
    y_pred = np.dot(tx_test, w)
    y_pred = np.where(y_pred<0.5,0,1)
    f1 = f1_score(y_test, y_pred)
    
    return loss, f1



def predictions(y, x, k_indices, k):
    """Returns the accuracies and f1-scores for each model.

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
        k:          scalar
    
    Returns:
        accuracies: a list of length 4 containing the accuracies for each model
        f1s: a list of length 4 containing the f1-scores for each model
    """

    # get k'th subgroup in test, others in train:
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    x_train = np.delete(x, k_indices[k], axis=0)
    y_train = np.delete(y, k_indices[k])

    # Change the labels to -1 and 1
    y_test = np.where(y_test==0,-1,1)

    
    # Logistic regression
    tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
    initial_w = np.zeros(tx_train.shape[1])
    w_lg, _ = logistic_regression(y_train, tx_train, initial_w, max_iters=300, gamma=0.01)

    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    y_pred_lg = np.dot(tx_test, w_lg)
    y_pred_lg = np.where(y_pred_lg<0.5,-1,1)

    # Regularized logistic regression
    tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
    initial_w = np.zeros(tx_train.shape[1])
    w_rlg, _ = reg_logistic_regression(y_train, tx_train, 0.001, initial_w, 300, 0.01)

    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    y_pred_rlg = np.dot(tx_test, w_rlg)
    y_pred_rlg = np.where(y_pred_rlg<0.5,-1,1)

    # Weighted regularized logistic regression
    tx_train = np.c_[np.ones((y_train.shape[0], 1)), x_train]
    initial_w = np.zeros(tx_train.shape[1])
    w_wrlg, _ = weighted_reg_logistic_regression(y_train, tx_train, 1e-7, initial_w, 300, 1.0)

    tx_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    y_pred_wrlg = np.dot(tx_test, w_wrlg)
    y_pred_wrlg = np.where(y_pred_wrlg<0.5,-1,1)
    

    # Accuracies and f1-scores
    accuracy_random = np.sum(y_test == np.random.choice([-1,1], size=y_test.shape[0])) / len(y_test)
    accuracy_lg = np.sum(y_pred_lg == y_test) / len(y_test)
    accuracy_rlg = np.sum(y_pred_rlg == y_test) / len(y_test)
    accuracy_wrlg = np.sum(y_pred_wrlg == y_test) / len(y_test)
  

    f1_random = f1_score(y_test, np.random.choice([-1,1], size=y_test.shape[0]))
    f1_lg = f1_score(y_test, y_pred_lg)
    f1_rlg = f1_score(y_test, y_pred_rlg)
    f1_wrlg = f1_score(y_test, y_pred_wrlg) 
    

    return [accuracy_random, accuracy_lg, accuracy_rlg, accuracy_wrlg], [f1_random, f1_lg, f1_rlg, f1_wrlg]