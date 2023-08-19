"""
Alexander Gutowsky, Ethan Saftler, Gatlin Newhouse, Manisha Katta, Rohan Reddy
CS 445/545 | Portland State University | ML Group Assignment Summer 2023
"""

from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
from utils.config import load_config, pretty_print_config
from utils.plot_learning import plot_metric
from utils.train_test_split import train_test_split
from utils.preprocess import balance_dataset


# Thanks to this tutorial for the base of the hyperparameter tuning code:
# https://www.tensorflow.org/tutorials/keras/keras_tuner
def model_builder(hp):
    """
    Builds a model with the given hyperparameters for use in hyperparameter tuning.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters to use in the model.

    Returns:
        model (tf.keras.Sequential): The model to use in hyperparameter tuning.
    """

    # Reading in configuration options
    try:
        config: dict = load_config(src="./config.yaml")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    input_shape = (18,)
    num_classes = 2
    model = tf.keras.Sequential()

    # Configure hyperparameter choices
    eta = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    alpha = hp.Choice("momentum", values=[1e-2, 1e-1, 2e-1, 5e-1, 7e-1, 9e-1])

    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=eta,
        momentum=alpha,
        nesterov=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGD",
    )

    # Hidden layer ranges
    hp_units = hp.Int(
        "hidden_units",
        min_value=int(config["params"]["hidden_units"]),
        max_value=101,
        step=10,
    )
    hl_units = hp.Int(
        "hidden_layers",
        min_value=int(config["params"]["hidden_layers"]),
        max_value=6,
        step=1,
    )

    # Get activation from config
    activation = hp.Choice("activation", values=["relu", "sigmoid", "tanh"])

    # Generating the hidden layers
    for _ in range(0, hl_units, 1):
        model.add(
            tf.keras.layers.Dense(
                units=hp_units,
                activation=activation,
                input_shape=input_shape,
            )
        )

    # Output Layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def main():
    num_classes = 2

    # Reading in configuration options
    print("- Attempting to read in config file from './config.yaml'")
    try:
        config: dict = load_config(src="./config.yaml")
        print(
            "Successfully obtained the following configuration options from './config.yaml'"
        )
        pretty_print_config(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "The specified config file does not exist. Is it located in './config.yaml'?"
        )
        exit(1)

    # Loading dataset into memory
    try:
        data: pd.DataFrame = pd.read_csv(config["datasets"]["dest"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "The specified CSV file does not exist. Is the file properly preprocessed?"
        )
        print("Usage: py main.py -p")
        exit(1)

    # X: Features
    # Y: Targets
    X, Y = balance_dataset(data)
    # X_train: Training features
    # X_test:  Testing features
    # Y_train: Training targets
    # Y_test:  Testing targets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, config["params"]["test_size"], config["params"]["seed"]
    )

    # Standardize the features using a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert labels to one-hot encoding
    Y_train_onehot = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_test_onehot = tf.keras.utils.to_categorical(Y_test, num_classes)

    print("Beginning Hyperparameter Tuning")

    # Number of models to train is:
    # 1 + log base factor of max_epochs and rounding it up to the nearest integer.
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=25,
        factor=3,
    )

    # Define early stopping callback based on validation loss
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    # Search for the best hyperparameters
    # 80% of the training data is used for training, 20% for validation
    tuner.search(
        X_train_scaled,
        Y_train_onehot,
        epochs=25,
        validation_split=0.2,
        callbacks=[stop_early],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

    # Print the optimal hyperparameters
    print(f"""The hyperparameter search is complete. The optimal number of units in {best_hps.get('hidden_layers')} layers
        is {best_hps.get('hidden_units')}. The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}
        and the optimal momentum is {best_hps.get('momentum')}. The optimal activation function for the hidden layers is
        {best_hps.get('activation')}.""")

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    # 80% of the training data is used for training, 20% for validation
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train_scaled, Y_train_onehot, epochs=50, validation_split=0.2)
    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))

    # Now we have our best hyperparameters and optimal number of epochs, we can train the model
    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel.fit(
        X_train_scaled, Y_train_onehot, epochs=best_epoch, validation_split=0.2
    )

    # Now print the accuracy and loss on the test set
    eval_result = hypermodel.evaluate(X_test_scaled, Y_test_onehot)
    print("[test loss, test accuracy]:", eval_result)

    # Print the optimal hyperparameters
    print("Optimal Hyperparameters:")
    print(f"- Number of hidden layers: {best_hps.get('hidden_layers')}")
    print(f"- Number of hidden units: {best_hps.get('hidden_units')}")
    print(f"- Activation function: {best_hps.get('activation')}")
    print(f"- Learning rate: {best_hps.get('learning_rate')}")
    print(f"- Momentum: {best_hps.get('momentum')}")
    print(f"- Best number of epochs to train: {best_epoch}")
    print("Done.")


if __name__ == "__main__":
    main()
