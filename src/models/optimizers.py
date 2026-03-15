class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
                p.data -= self.lr * p.grad

class Adam: # ini kalo mw implement adam
    # mas adam~ mas adam~

    # btw implement Adam ya, jangan AdamW
    pass