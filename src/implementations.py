'''
 File name: implementations.py
 Author: ML Wizards
 Date created: 06/10/2023
 Date last modified: 08/10/2023
 Python Version: 3.9.18
 '''
import numpy as np
from utility import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape=(N, ), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> mean_squared_error_gd(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.5,0.5]), 50, 0.1)
    array([ 0.18265102, -0.09854342]), 0.00020163687806323794
    """
    w = initial_w

    for n_iter in range(max_iters):

        gradient = compute_gradient(y,tx,w)
        w = w - gamma * gradient

    mse = compute_loss(y,tx,w)
    return w, mse

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.

    Args:
        y: numpy array of shape=(N, ), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> mean_squared_error_sgd(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), np.array([0.5,0.5]), 50, 0.1)
    array([ 0.19153232, -0.10243375]), 0.0002652216422081275
    """

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx):

            gradient = compute_gradient(minibatch_y,minibatch_tx,w)
            w = w - gamma * gradient

    mse = compute_loss(y,tx,w)
    return w, mse

def least_squares(y, tx):
    """Least squares regression using normal equations.
    Calculates the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    array([ 0.21212121, -0.12121212]), 8.666684749742561e-33
    """

    w = np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    e = y - np.dot(tx,w)
    mse = 0.5 * np.mean(e*e)

    return w, mse

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212]), 8.666684749742561e-33
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628]), 0.006417022643777357
    """

    w = np.linalg.solve((np.dot(tx.T,tx)+lambda_*2*tx.shape[0]*np.identity(tx.shape[1])), np.dot(tx.T,y))
    mse = compute_loss(y,tx,w)

    return w, mse

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent
    
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w -= grad * gamma

    loss = compute_loss_neg_log_likelihood(y, tx, w)

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: a scalar denoting the regularization parameter
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w -= grad * gamma

    loss = compute_loss_neg_log_likelihood(y, tx, w)

    return w, loss


def weighted_reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Weighted regularized logistic regression using gradient descent
    
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: a scalar denoting the regularization parameter
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        loss: scalar.
    """

    # The weights
    class_1_samples = np.sum(y)
    class_0_samples = len(y) - class_1_samples
    weight_0 = len(y) / (2 * class_0_samples)
    weight_1 = len(y) / (2 * class_1_samples)
    class_weights = {0: weight_0, 1: weight_1}

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_weighted_gradient_logistic(y, tx, w, class_weights) + 2 * lambda_ * w
        w -= grad * gamma

    loss = compute_weighted_loss_neg_log_likelihood(y, tx, w, class_weights)

    return w, loss

def learning_by_weighted_penalized_gradient(y_tr, tx_tr, w, class_weights, lambda_, gamma):
    """One iteration of weighted regularized logistic regression using gradient descent
    
    Args: 
        y_tr: numpy array of shape (N,), N is the number of samples.
        tx_tr: numpy array of shape (N, D), D is the number of features.
        class_weights: dict with weights for class 1 and 0
        lambda_: a scalar denoting the regularization parameter
        gamma: a scalar denoting the stepsize

    Returns:
        w: new weights, numpy array of shape(D,).
        loss: scalar.
    """
    
    loss = compute_weighted_loss_neg_log_likelihood(y_tr, tx_tr, w, class_weights) + lambda_ * np.linalg.norm(w)**2
    gradient = compute_weighted_gradient_logistic(y_tr, tx_tr, w, class_weights) + 2 * lambda_ * w
    w -= gamma * gradient

    return loss, w

def weighted_reg_logistic_regression_demo(y_tr, tx_tr, tx_val, y_val, initial_w, max_iter, lambda_= 1e-7, gamma=1.0):
    """Weighted regularized logistic regression using gradient descent and calculating the best model
    
    Args: 
        y_tr: numpy array of shape (N,), N is the number of training samples.
        tx_tr: numpy array of shape (N, D), D is the number of training features.
        y_val: numpy array of shape (N,), N is the number of validation samples.
        tx_val: numpy array of shape (N, D), D is the number of validation features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        lambda_: a scalar denoting the regularization parameter
        gamma: a scalar denoting the stepsize

    Returns:
        losses: numpy array of shape (max_iter,)
        scores: numpy array of shape (max_iter,)
        val_losses: numpy array of shape (max_iter,)
        val_scores: numpy array of shape (max_iter,)
        best_w: numpy array of shape (D,), the best weights in terms of F1 validation score
        best_f1: the best validation score
    """
    
    # init parameters
    w = initial_w
    losses = []
    scores = []
    val_losses = []
    val_scores = []

    best_w = np.zeros(tx_val.shape[1])
    best_f1 = 0

    # The weights
    class_1_samples = np.sum(y_tr)
    class_0_samples = len(y_tr) - class_1_samples
    weight_0 = len(y_tr) / (2 * class_0_samples)
    weight_1 = len(y_tr) / (2 * class_1_samples)
    class_weights = {0: weight_0, 1: weight_1}

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_weighted_penalized_gradient(y_tr, tx_tr, w, class_weights, lambda_, gamma)

        losses.append(loss)

        y_pred_tr = np.dot(tx_tr, w)
        y_pred_tr = np.where(y_pred_tr > 0.5, 1, 0)
        f1_tr = f1_score(y_tr, y_pred_tr)
        scores.append(f1_tr)

        val_loss = compute_weighted_loss_neg_log_likelihood(y_val, tx_val, w, class_weights) + lambda_ * np.linalg.norm(w)**2
        val_losses.append(val_loss)
        y_pred_val = np.dot(tx_val, w)
        y_pred_val = np.where(y_pred_val > 0.5, 1, 0)
        f1_val = f1_score(y_val, y_pred_val)
        val_scores.append(f1_val)

        if f1_val > best_f1:
            best_w = w.copy()
            best_f1 = f1_val

    return losses, scores, val_losses, val_scores, best_w, best_f1