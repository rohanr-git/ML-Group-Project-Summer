import numpy as np


def train_test_split(X, Y, test_size, random_state):
    np.random.seed(random_state)
    indices = np.random.permutation(X.shape[0])
    test_count = int(test_size * X.shape[0])
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test
