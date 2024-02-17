'''
 File name: feature_engineering.py
 Author: ML Wizards
 Date created: 29/10/2023
 Date last modified: 29/10/2023
 Python Version: 3.9.18
 '''

import numpy as np

def best_features(x_train, y_train, nb_features):
    """Calculates the most correlated features with y_train

    Args:
        x_train: the training data
        y_train: the training labels
        nb_features: the number of features to keep

    Returns:
        the indices of the best features
    """
    
    # Calculate the correlation between each column of x_train and y_train
    correlations = np.corrcoef(x_train.T, y_train)

    # Extract the correlation coefficients for each feature
    correlation_with_y = correlations[-1, :-1]

    # Sort features by absolute correlation in descending order
    sorted_indices = np.argsort(np.abs(correlation_with_y))[::-1]
    
    return sorted_indices[:nb_features]


def one_hot_encoding(feature):
    """Performs one-hot encoding on a feature

    Args:
        feature: numpy array of shape=(N, ), N is the number of samples.
    
    Returns:
        numpy array of shape=(N, K), K is the number of unique values in the feature.
    """

    # Get unique values in the feature
    unique_values = np.unique(feature)

    # Initialize a matrix of zeros to represent the one-hot encoding
    one_hot_encoding = np.zeros((len(feature), len(unique_values)), dtype=float)

    # Iterate through the feature values and set the corresponding one-hot encoding
    for i, value in enumerate(feature):
        one_hot_encoding[i, np.where(unique_values == value)[0]] = 1

    return one_hot_encoding


def categorical_features_processing(x_train, indices):
    """Processes the categorical features with one-hot encoding

    Args:
        x_train: numpy array of shape=(N,D), D is the number of features.
        indices: list of indices of the categorical features to process

    Returns:
        nupmy array of shape=(N, K), K is the number of unique values in the features.

    """

    # Encode each categorical feature
    list_encoded = []
    for idx in indices:
        list_encoded.append(one_hot_encoding(x_train[:,idx]))
    
    # Concatenate the encoded features
    features = np.hstack(list_encoded)

    return features

def standardize(x):
    """ Standardize the data set.

    Args:
        x: numpy array of shape=(N,D), D is the number of features.
    
    Returns:
        numpy array of shape=(N,D), D is the number of features.
    """

    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x


def PCA(x_train, x_test, k):
    """Performs PCA on the training and test data

    Args:
        x_train: numpy array of shape=(N,D), D is the number of features.
        x_test: numpy array of shape=(N,D), D is the number of features.
        k: number of principal components to keep
    
    Returns:
        train_pca: numpy array of shape=(N,K), K is the number of principal components.
        test_pca: numpy array of shape=(N,K), K is the number of principal components.
    """

    # Center the data by subtracting the mean of each feature
    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_train, rowvar=False)

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Choose the top k eigenvectors based on the explained variance
    top_k_eigenvectors = eigenvectors[:, :k]

    # Project the training data onto the selected principal components
    train_pca = np.real(centered_train.dot(top_k_eigenvectors))

    # Center and transform the test data
    centered_test = x_test - mean
    test_pca = np.real(centered_test.dot(top_k_eigenvectors))

    return train_pca, test_pca

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to validation.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_val: numpy array containing the validation data.
        y_tr: numpy array containing the train labels.
        y_val: numpy array containing the validation labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
 
    idx = np.random.permutation(len(x))
    split = int(np.floor(len(x) * ratio))
    idx_tr = idx[:split]
    idx_val = idx[split:]
    
    x_tr = x[idx_tr]
    x_val = x[idx_val]
    y_tr = y[idx_tr]
    y_val = y[idx_val]    
    
    return x_tr, x_val, y_tr, y_val

