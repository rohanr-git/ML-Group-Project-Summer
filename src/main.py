'''
Alexander Gutowsky, Ethan Saftler, Gatlin Newhouse, Manisha Katta, Rohan Reddy
CS 445/545 | Portland State University | ML Group Assignment Summer 2023
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from utils.preprocess import preprocess
from utils.config import load_config

def main():
    config = load_config("../config.yaml")
    preprocess(src=config["datasets"]["src"], dest=config["datasets"]["dest"])
    data = pd.read_csv(config["datasets"]["dest"], header=None, low_memory=False)
    X = data.iloc[:, :-1]  # All features of the dataset (columns 0-17)
    Y = data.iloc[:, -1]   # All labels of the dataset (column 18)
    # The data is split into training and testing sets (60% training, 40% testing)
    # X_train: Training features
    # X_test:  Testing features
    # Y_train: Training targets
    # Y_test:  Testing targets
    # random_state is basically the seed - use the same value for reproducible results
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=config["seeds"]["random_state"])

if __name__ == '__main__':
    main()
