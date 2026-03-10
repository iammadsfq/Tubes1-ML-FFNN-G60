import numpy as np

class Tensor:
    """ Menyimpan nilai skalar/matriks dan gradiennya untuk Autodiff. """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            
            while grad_self.ndim > self.data.ndim:
                grad_self = np.sum(grad_self, axis=0)
            for i in range(self.data.ndim):
                if self.data.shape[i] == 1 and grad_self.shape[i] > 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)
            
            while grad_other.ndim > other.data.ndim:
                grad_other = np.sum(grad_other, axis=0)
            for i in range(other.data.ndim):
                if other.data.shape[i] == 1 and grad_other.shape[i] > 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)
            
            self.grad += grad_self
            other.grad += grad_other
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out


    def backward(self):
        """ Memulai proses backpropagation dari node ini. """
        pass