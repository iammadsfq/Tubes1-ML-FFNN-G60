from abc import ABC, abstractmethod

# aku cinta abstraksi
class BaseOptimizer(ABC):
     @abstractmethod
     def step(self) -> None:
          pass

class SGD(BaseOptimizer):
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
                p.data -= self.lr * p.grad

class Adam(BaseOptimizer): # ini kalo mw implement adam

    # mas adam~ mas adam~
    # btw implement Adam ya, jangan AdamW

    def step(self) -> None:
         pass