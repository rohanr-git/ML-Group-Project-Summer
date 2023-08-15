'''
Alexander Gutowsky, Ethan Saftler, Gatlin Newhouse, Manisha Katta, Rohan Reddy
CS 445/545 | Portland State University | ML Group Assignment Summer 2023
'''

import pandas as pd
import numpy as np

# Hyperparameters
MOMENTUM = 0.01
LEARNING_RATE = 0.01
EPOCHS = 50
HIDDEN_SIZE = 21
TEST_SIZE = 0.4

def sig(Z):
    return 1 / (1 + np.exp(-Z))

def sig_deriv(Z):
    return sig(Z) * (1 - sig(Z))

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def main():
    # Load data
    data = pd.read_csv("datasets/CVD_processed.csv") # Adjust the path to your CSV file
    data.iloc[:, 8:12] = data.iloc[:, 8:12] / 500

    # Balance data due to extreme class imbalance (92% negative, 8% positive)
    data_majority = data[data.iloc[:, -1] == 0]
    data_minority = data[data.iloc[:, -1] == 1]
    # Oversample minority class
    minority_size = data_minority.shape[0]
    majority_size = data_majority.shape[0]
    # Repeat the minority class rows until it matches the size of the majority class
    oversampled_minority = data_minority.loc[data_minority.index.repeat((majority_size // minority_size) + 1)].reset_index(drop=True)
    oversampled_minority = oversampled_minority.iloc[:majority_size, :]
    # Combine majority class with oversampled minority class
    data_balanced = pd.concat([data_majority, oversampled_minority])
    # Shuffle the data
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data_balanced.iloc[:, :-1].values
    Y = data_balanced.iloc[:, -1].values


    # Split data
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    test_count = int(TEST_SIZE * X.shape[0])
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    # Calculate the mean and standard deviation for each feature in the training set
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Normalize the training data
    X_train = (X_train - mean) / std

    # Normalize the test data using the same mean and standard deviation
    X_test = (X_test - mean) / std

    # Initialize weights and biases
    input_size = X_train.shape[1]
    W1 = np.random.uniform(-0.5, 0.5, (input_size, HIDDEN_SIZE))
    b1 = np.zeros((1, HIDDEN_SIZE))
    W2 = np.random.uniform(-0.5, 0.5, (HIDDEN_SIZE, 1))
    b2 = np.zeros((1, 1))


    # Training
    for epoch in range(EPOCHS):
        # Forward pass
        Z1 = np.dot(X_train, W1) + b1
        A1 = sig(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sig(Z2)

        # Backward pass
        dZ2 = A2 - Y_train.reshape(-1, 1)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, W2.T) * sig_deriv(Z1)
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
    A1 = sig(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sig(Z2)
    predictions = (A2 > 0.5).astype(int)
    cm = confusion_matrix(Y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    main()