'''
 File name: utility.py
 Author: ML Wizards
 Date created: 08/10/2023
 Date last modified: 08/10/2023
 Python Version: 3.9.18
 '''
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, ), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    return (1/(2*y.shape[0]))*np.sum((y-np.dot(tx,w))**2)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, ), N is the number of samples.
        tx: shape=(N,D), D is the number of features.
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ), containing the gradient of the loss at w.
    """

    return (-1/y.shape[0])*np.dot(tx.T,y-np.dot(tx,w))

def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """Computes the sigmoid value of t.
    
    Args:
        t: scalar or numpy array.
    
    Returns:
        scalar or numpy array.
    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    
    epsilon = 1e-15
    sigmoid = 1.0/(1.0 + np.exp(-t))
    sigmoid = np.clip(sigmoid, epsilon, 1 - epsilon)
    return sigmoid

def compute_loss_neg_log_likelihood(y, tx, w):
    """Computes the loss with negative log likelihood.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """

    y_pred = sigmoid(np.dot(tx, w))
    loss = np.dot(y.T, np.log(y_pred)) + np.dot((1-y).T, np.log(1-y_pred))
    return np.squeeze(-loss) * (1/y.shape[0])

def compute_gradient_logistic(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N,), N is the number of samples.
        tx: shape=(N, D), D is the number of features.
        w: shape=(D,). The vector of model parameters.

    Returns:
        A vector of shape (D,), containing the gradient of the loss at w.
    """

    y_pred = sigmoid(np.dot(tx, w))
    return np.dot(tx.T, (y_pred - y)) * (1/y.shape[0])


def compute_weighted_loss_neg_log_likelihood(y, tx, w, class_weights):
    """Computes the weighted loss with negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a loss value.
    """
    y_pred = sigmoid(np.dot(tx, w))
    
    loss = class_weights[1]*np.dot(y.T, np.log(y_pred)) + class_weights[0]*np.dot((1-y).T, np.log(1-y_pred))
    loss = np.squeeze(-loss).item() * (1/y.shape[0])
    return loss


def compute_weighted_gradient_logistic(y, tx, w, class_weights):
    """Computes the gradient at w with weights.

    Args:
        y: shape=(N, 1), N is the number of samples.
        tx: shape=(N, D), D is the number of features.
        w: shape=(D, 1). The vector of model parameters.

    Returns:
        A vector of shape (D, 1), containing the gradient of the loss at w.
    """

    y_pred = sigmoid(np.dot(tx, w))
    gradient = (1/y.shape[0]) * np.dot(tx.T, (class_weights[0]*(1-y)*y_pred - class_weights[1]*y*(1-y_pred)))
    return gradient

def f1_score(y,y_pred):
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = sum(1 for a, b in zip(y, y_pred) if a == 1 and b == 1)
    FP = sum(1 for a, b in zip(y, y_pred) if a == 0 and b == 1)
    FN = sum(1 for a, b in zip(y, y_pred) if a == 1 and b == 0)

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1
