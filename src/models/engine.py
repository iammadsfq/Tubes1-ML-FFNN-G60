import numpy as np

class Tensor:
    """ Menyimpan nilai skalar/matriks dan gradiennya untuk Autodiff. """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad *= self.data.T @ out.grad
        
        out._backward = _backward
        return out


    def backward(self):
        """ Memulai proses backpropagation dari node ini. """
        pass