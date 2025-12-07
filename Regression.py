import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

def get_mini_batches(data_len: int, batch_size: int = 32):
    """ Generates mini-batches based on the data indexes
        
        Args:
            data_len: Length of the data
            
            batch_size: Size of each mini batch where the last mini-batch
                might be smaller than the rest if the batch_size does not 
                evenly divide the data length.

    """
    X_idx = np.arange(data_len)
    np.random.shuffle(X_idx)
    batches = [X_idx[i:i+batch_size] for i in range(0, data_len, batch_size)]

    return batches

def mse(y_hat, y):
    return np.mean(pow((y_hat-y), 2))
def rmse(y_hat,y):
    return np.sqrt(mse(y_hat,y))