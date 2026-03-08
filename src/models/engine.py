import numpy as np

class Value:
    """ Menyimpan nilai skalar/matriks dan gradiennya untuk Autodiff. """
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        def _backward():
            pass
        pass

    def __mul__(self, other):
        pass

    def backward(self):
        """ Memulai proses backpropagation dari node ini. """
        pass