import numpy as np
from models.engine import Tensor
from abc import ABC, abstractmethod

class BaseActivation(ABC):
    """Do FF by calling the instance"""
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

class Linear(BaseActivation):
    """Linear(x) = x"""
    def forward(self, x):
        out = Tensor(x.data, (x,), 'linear')

        def _backward():
            x.grad += 1.0*out.grad

        out._backward = _backward
        return out

class ReLU(BaseActivation):
    """ReLU(x) = max(0, x)"""
    def forward(self, x):
        out = Tensor(np.maximum(0, x.data), (x,), 'ReLU')

        def _backward():
            x.grad += (x.data > 0).astype(float)*out.grad # 1 kalo x > 0; else 0

        out._backward = _backward
        return out

class Sigmoid(BaseActivation):
    """Sigmoid(x) = 1/(1+exp(-x))"""
    def forward(self, x):

        sig = np.where(x.data >= 0, 1 / (1 + np.exp(-x.data)), np.exp(x.data)/ (1 + np.exp(x.data)))

        out = Tensor(sig, (x,), 'Sigmoid')
        def _backward():
            x.grad += (sig * (1.0 - sig)) * out.grad

        out._backward = _backward
        return out

class TanH(BaseActivation):
    """tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))"""
    def forward(self, x):
        e1 = np.exp(x.data)
        e2 = np.exp(-x.data)

        v = (e1-e2)/(e1+e2)
        out = Tensor(v, (x,), 'Tanh')

        def _backward():
            x.grad += (1.0 - v**2) * out.grad

        out._backward = _backward
        return out

class Softmax(BaseActivation):
    """
    Softmax(x)_i = exp(x_i) / sum(exp(x_j))
    Dapat menerima input berupa batch
    Refrensi: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    def forward(self, x):
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

class LeakyReLU(BaseActivation):
    """Leaky ReLU(x) = ax if x <= 0 else x"""
    def __init__(self, a=0.01):
        super().__init__()
        self.a = a

    def forward(self, x):
        out = Tensor( np.where(x.data <= 0, self.a*x.data, x.data), (x,), 'Leaky ReLU')

        def _backward():
            x.grad += np.where(x.data <= 0, self.a, 1) * out.grad

        out._backward = _backward
        return out
