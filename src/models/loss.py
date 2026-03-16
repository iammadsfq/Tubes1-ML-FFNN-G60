import numpy as np
from models.engine import Tensor
from abc import ABC, abstractmethod

class BaseLoss(ABC):
    """Calc loss by calling the instance"""
    @abstractmethod
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, y_true: Tensor, y_pred: Tensor):
        return self.forward(y_true, y_pred)

class MSELoss(BaseLoss):
    """
    MSE = 1/n * sum((y_true-y_pred)^2)
    n = batch size
    """
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        assert y_true.data.shape == y_pred.data.shape, \
            f"Shape mismatch: y_true {y_true.data.shape} vs y_pred {y_pred.data.shape}"

        n = y_true.data.shape[0]
        diff = y_true.data - y_pred.data

        loss_val = np.sum(np.square(diff)) / n

        out = Tensor(loss_val, (y_pred,), 'MSE')

        def _backward():
            y_pred.grad += (2.0/n)*diff*out.grad

        out._backward = _backward
        return out

class BinaryClassEntropy(BaseLoss):
    """
    Menghitung Binary Cross-Entropy untuk klasifikasi biner
    Menggunakan ln
    """
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        assert y_true.data.shape == y_pred.data.shape, \
            f"Shape mismatch: y_true {y_true.data.shape} vs y_pred {y_pred.data.shape}"

        n = y_true.data.shape[0] # batch size

        # avoid log(0)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)

        loss_val = -np.mean(y_true.data * np.log(y_pred_clipped) + (1 - y_true.data) * np.log(1 - y_pred_clipped))

        out = Tensor(loss_val, (y_pred,), 'BCE')

        def _backward():
            grad = (y_pred_clipped - y_true.data) / (y_pred_clipped * (1.0 - y_pred_clipped))
            y_pred.grad += (1.0 / n) * grad * out.grad

        out._backward = _backward
        return out

class CategoricalClassEntropy(BaseLoss):
    """
    Menghitung Categorical Cross-Entropy untuk multi-class
    y_true diasumsikan dalam bentuk one-hot encoding
    """
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        assert y_true.data.shape == y_pred.data.shape, \
            f"Shape mismatch: y_true {y_true.data.shape} vs y_pred {y_pred.data.shape}"

        n = y_true.data.shape[0] # Batch size

        # avoid log(0)
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)

        loss_val = -np.sum(y_true.data * np.log(y_pred_clipped)) / n

        out = Tensor(loss_val, (y_pred,), 'CCE')

        def _backward():
            grad = -(y_true.data / y_pred_clipped)
            y_pred.grad += (1.0/n) * grad * out.grad

        out._backward = _backward
        return out