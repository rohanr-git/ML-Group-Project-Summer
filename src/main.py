'''
Alexander Gutowsky, Ethan Saftler, Gatlin Newhouse, Manisha Katta, Rohan Reddy
CS 445/545 | Portland State University | ML Group Assignment Summer 2023
'''

import pandas as pd
import utils.mathfuncs as mf
import numpy as np
from utils.config import load_config
from utils.train_test_split import train_test_split
from utils.preprocess import preprocess_csv, balance_dataset


def main():
    # Load configuration options
    config = load_config("config.yaml")

    # Preprocess and load data into memory
    data = preprocess_csv(src=config["datasets"]["src"], dest=config["datasets"]["dest"])

    # Load data
    data = pd.read_csv(config["datasets"]["dest"])

    # X: Features
    # Y: Targets
    X, Y = balance_dataset(data)

    # X_train: Training features
    # X_test:  Testing features
    # Y_train: Training targets
    # Y_test:  Testing targets

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, config["model_1"]["test_size"], config["model_1"]["seed"])

    # Calculate the mean and standard deviation for each feature in the training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Normalize the training data
    X_train = mf.normalize(X_train, mean, std)

    # Normalize the test data using the same mean and standard deviation
    X_test = mf.normalize(X_test, mean, std)

    # Initialize weights and biases
    input_size = X_train.shape[1]
    W1, b1, W2, b2 = mf.initalize_random_weights(input_size, config["model_1"]["hidden_size"])

    # Hyperparameters
    MOMENTUM = config["model_1"]["momentum"]
    LEARNING_RATE = config["model_1"]["learning_rate"]
    EPOCHS = config["model_1"]["epochs"]

    # Training
    for epoch in range(EPOCHS):
        # Forward pass
        Z1 = np.dot(X_train, W1) + b1
        A1 = mf.sig(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = mf.sig(Z2)

        # Backward pass
        dZ2 = A2 - Y_train.reshape(-1, 1)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, W2.T) * mf.sig_deriv(Z1)
        dW1 = np.dot(X_train.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update with momentum
        W2 = W2 - LEARNING_RATE * dW2 + MOMENTUM * W2
        b2 = b2 - LEARNING_RATE * db2 + MOMENTUM * b2
        W1 = W1 - LEARNING_RATE * dW1 + MOMENTUM * W1
        b1 = b1 - LEARNING_RATE * db1 + MOMENTUM * b1

        # Accuracy
        predictions = (A2 > 0.5).astype(int)
        accuracy = np.mean(predictions == Y_train.reshape(-1, 1))
        print(f"Epoch {epoch + 1}: Accuracy {accuracy * 100:.2f}%")

    # Test
    Z1 = np.dot(X_test, W1) + b1
    A1 = mf.sig(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = mf.sig(Z2)
    predictions = (A2 > 0.5).astype(int)
    cm = mf.confusion_matrix(Y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()
