import matplotlib.pyplot as plt
from tensorflow import keras

def plot_metric(history, metric):
    if "accuracy" in metric:
        plt.title("Accuracy per Epoch %")
        plt.ylabel("Accuracy %")
    else:
        plt.title("Loss per Epoch")
        plt.ylabel("Loss")

    # Label the x-axis
    plt.xlabel("Epoch")

    # Plot the graph
    plt.plot(history[metric])
    plt.plot(history[f"val_{metric}"])

    # Add a legend
    plt.legend(["train", "test"], loc="upper left")

    # Save the graph
    plt.savefig(f"graphs/epoch_{metric}.png")
    print(f"Saved graph of {metric} per epoch to graphs/epoch_{metric}.png")
