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

    # clamp x.data? idk it works so far lol
    sig = 1.0 / (1.0 + np.exp(-x.data))

    out = Tensor(sig, (x,), 'Sigmoid')
    def _backward():
        x.grad += (sig * (1.0 - sig)) * out.grad

    out._backward = _backward
    return out

def tanh(x):
    """tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))"""
    e1 = np.exp(x.data)
    e2 = np.exp(-x.data)

    v = (e1-e2)/(e1+e2)
    out = Tensor(v, (x,), 'Tanh')

    def _backward():
        x.grad += ((2.0/(e1-e2)) ** 2) * out.grad

    out._backward = _backward
    return out

def softmax(x):
    """
    Softmax(x)_i = exp(x_i) / sum(exp(x_j))
    Dapat menerima input berupa batch
    Refrensi: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    x_shifted = x.data - np.max(x.data, axis=1, keepdims=True) #shifted untuk numerical stability
    exp_x = np.exp(x_shifted)
    sm = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    out = Tensor(sm, (x,), 'Softmax')
    def _backward():
        grad = np.zeros_like(sm)
        for i in range(len(sm)):
            s_i = np.reshape(sm[i], (-1, 1))
            jm = np.diagflat(s_i) - (np.dot(s_i, s_i.T))
            grad[i] = np.dot(jm, out.grad[i])
        x.grad += grad
            
    out._backward = _backward
    return out


## BONUS

def leakyrelu(x, a=0.01):
    """Leaky ReLU(x) = ax if x <= 0 else x"""
    out = Tensor( np.where(x.data <= 0, a*x.data, x.data), (x,), 'Leaky ReLU')

    def _backward():
        x.grad += np.where(x.data <= 0, a, 1) * out.grad

    out._backward = _backward
    return out
