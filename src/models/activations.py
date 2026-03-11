import numpy as np
from models.engine import Tensor

def linear(x):
    """Linear(x) = x"""
    out = Tensor(x.data, (x,), 'linear')
    
    def _backward():
        x.grad += 1.0*out.grad

    out._backward = _backward
    return out