import numpy as np

def sig(Z) : 
    '''
    Calculate the sigmoid activation function for a scalar value.

    Args:
        Z (float): A scalar input value.

    Returns:
        float: The output of the sigmoid function for the given input.
    '''
    return 1 / (1 + np.exp(-Z))
