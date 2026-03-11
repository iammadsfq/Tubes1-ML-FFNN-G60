import numpy as np
from models.engine import Tensor


def mse_loss(y_true, y_pred):
    """
    MSE = 1/n * sum((y_true-y_pred)^2)
    n = batch size
    """
    assert y_true.shape == y_pred.shape, \
        f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    
    n = y_true.shape[0]
    diff = y_true.data - y_pred.data
    
    loss_val = np.sum(np.square(diff)) / n

    out = Tensor(loss_val, (y_pred,), 'MSE')

    def _backward():
        y_pred.grad += (2.0/n)*diff*out.grad
    
    out._backward = _backward
    return out