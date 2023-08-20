import matplotlib.pyplot as plt
from utils.config import load_config


def plot_metric(history, metric, hypermodel=False):
    """
    Plot a graph of the metric vs. epochs.

    Args:
        history (dict): The history object returned by model.fit()
        metric (str): The metric to plot
    """
    # Clear the current figure
    plt.clf()

    # Load the config values
    config: dict = load_config(src="./config.yaml")
    epochs = config["params"]["epochs"]
    learningrate = config["params"]["learning_rate"]
    momentum = config["params"]["momentum"]
    hiddenlayers = config["params"]["hidden_layers"]
    hiddenunits = config["params"]["hidden_units"]
    batchsize = config["params"]["batch_size"]
    activation = config["params"]["activation"]

    # Set the title and y-axis label
    if "accuracy" in metric:
        plt.title("Accuracy per Epoch %")
        plt.ylabel("Accuracy %")
    else:
        plt.title("Loss per Epoch")
        plt.ylabel("Loss")

    # Label the x-axis
    plt.xlabel("Epoch")

    # Plot the graph
    plt.plot(history.history[metric])
    plt.plot(history.history[f"val_{metric}"])

    # Add a legend
    plt.legend(["train", "test"], loc="upper left")

    # Save the graph
    if hypermodel:
        plt.savefig(
            f"graphs/hypermodel/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
        print(
            f"Saved graph of {metric} per epoch to graphs/hypermodel/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
    else:
        plt.savefig(
            f"graphs/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
        print(
            f"Saved graph of {metric} per epoch to graphs/{activation}/{metric}_e{epochs}_lr{learningrate}_m{momentum}_hl{hiddenlayers}_hu{hiddenunits}_bs{batchsize}.png"
        )
