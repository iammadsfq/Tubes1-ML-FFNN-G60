import numpy as np
from models.engine import Tensor

def linear(x):
    """Linear(x) = x"""
    out = Tensor(x.data, (x,), 'linear')
    
    def _backward():
        x.grad += 1.0*out.grad

    out._backward = _backward
    return out

def relu(x):
    """ReLU(x) = max(0, x)"""
    out = Tensor(np.maximum(0, x.data), (x,), 'ReLU')

    def _backward():
        x.grad += (x.data > 0).astype(float)*out.grad # 1 kalo x > 0; else 0
    
    out._backward = _backward
    return out

def sigmoid(x):
    """Sigmoid(x) = 1/(1+exp(-x))"""
    def _backward():
        pass
    pass

def tanh(x):
    """tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))"""
    def _backward():
        pass
    pass

def softmax(x):
    """
    Softmax(x)_i = exp(x_i) / sum(exp(x_j))
    Dapat menerima input berupa batch
    """
    def _backward():
        pass
    pass