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

def sig_deriv(Z):
    '''
    

    Args:
        

    Returns:
        
    '''
    return sig(Z) * (1 - sig(Z))

def confusion_matrix(y_true, y_pred):
    '''
    
    Args:
        

    Returns:
        
    '''
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def normalize(X, mean, std):
    '''
    
    Args:
        

    Returns:
        
    '''
    return (X - mean) / std

def initalize_random_weights(input_size, hidden_size):
    '''
    
    '''
    W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
    b2 = np.zeros((1, 1))

    return W1, b1, W2, b2