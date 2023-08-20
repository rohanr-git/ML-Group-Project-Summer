# ML-Group-Project-Summer-2023

This is a group research project that compares the performance and accuracies of two [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) networks on the [cardiovascular diseases risk prediction dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset).

## Getting Started

### Prerequisites
- Have a valid [GitHub account](https://github.com/join), Install [Git](https://git-scm.com/), [generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent), and [add this SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
- Install the latest version of [Python](https://www.python.org/downloads/). This will come preinstalled with [pip](https://pip.pypa.io/en/stable/).

### Installation

1. Clone this repository, or fork your own:
    ``git clone git@github.com:agutowsky/ML-Group-Project-Summer-2023.git``
    > How to fork your own repository: https://docs.github.com/en/get-started/quickstart/fork-a-repo

2. Install the required python packages with:
    ``pip install -r requirements.txt``
    > Note: It is recommended to install packages within a virtual environment to avoid conflicts with other project dependencies.
    Read more here: https://code.visualstudio.com/docs/python/environments

3.  (Optional) optimize hyperparameters:
    ``py src/tune_hps.py``

4.  Run the project:
    ``py src/main.py``

## License

This project is distributed under the MIT License.

See ``LICENSE.txt`` for more information, or read the official document [here](https://opensource.org/license/mit/).

## Disclaimers
This project is for machine learning research and educational purposes only. The machine learning models developed for this project should not be used as a substitute for professional medical advice or diagnosis. If you have concerns about your cardiovascular health, please consult a qualified healthcare professional.