from .engine import Tensor
from models.optimizers import *
import models.activations as activations
import models.loss as loss_module
import numpy as np

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Layer(Module):
    def __init__(
            self,
            nin,
            nout,
            activation='linear',
            init_method='normal',
            random_state=42,
            **kwargs):
        if random_state is not None:
            np.random.seed(random_state)
        self.activation_name = activation.lower()

        if init_method == 'zero':
            w_data = np.zeros((nin, nout))
        elif init_method == 'uniform':
            low, high = kwargs.get('lower', -0.1), kwargs.get('upper', 0.1)
            w_data = np.random.uniform(low, high, (nin, nout))
        elif init_method == 'normal':
            mean, var = kwargs.get('mean', 0.0), kwargs.get('variance', 1.0)
            w_data = np.random.normal(mean, np.sqrt(var), (nin,nout))
        elif init_method == 'xavier':
            w_data = np.random.randn(nin, nout) * np.sqrt(1/nin)
        elif init_method == 'he':
            w_data = np.random.randn(nin, nout) * np.sqrt(2/nin)

        self.w = Tensor(w_data)
        self.b = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        act_func = getattr(activations, self.activation_name)
        return act_func(x @ self.w + self.b)

    def parameters(self):
        return [self.w, self.b]

class FFNN(Module):
    def __init__(
        self,
        layers_config,
        activations='relu', # bisa string atau array of strings
        loss_function='mse',
        init_method='normal',
        random_state=42
    ):
        self.layers_config = layers_config
        self.loss_name = loss_function
        self.layers = []

        if isinstance(activations, str):
            activations = [activations] * (len(layers_config) - 1)

        for i in range(len(layers_config) - 1):
            nin = layers_config[i]
            nout = layers_config[i+1]
            act = activations[i]
            layer_seed = random_state + i if random_state is not None else None
            self.layers.append(Layer(nin, nout, activation=act, init_method=init_method, random_state=layer_seed))

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            solver='sgd',
            batch_size=32,
            lr=0.01,
            epochs=100,
            verbose=1
            ):

        history = {'train_loss': [], 'val_loss': []}

        # [x] forward pass
        # [x] calc loss
        # [ ] regularisation (L1/L2)
        # [x] backward pass
        # [x] optimizer step
        # [x] append loss history
        # [ ] progress bar?

        if solver.lower() == 'sgd':
            optimizer: BaseOptimizer = SGD(self.parameters(), lr=lr)
        elif solver.lower() == 'adam':
            optimizer: BaseOptimizer = Adam()
        else:
            raise ValueError(f"{solver} opitimzer is not supported yet.")

        if self.loss_name.lower() == 'mse':
            loss_fn: callable = loss_module.mse_loss
        elif self.loss_name.lower() == 'bce':
            loss_fn: callable = loss_module.binary_cross_entropy
        else:
            raise ValueError(f"Loss {self.loss_name} not supported yet.")

        history = {'train_loss': [], 'val_loss': []}
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = Tensor(X_train[batch_indices])
                y_batch = Tensor(y_train[batch_indices])

                y_pred = self.forward(X_batch)

                loss = loss_fn(y_batch, y_pred)
                epoch_loss += loss.data
                batches += 1

                self.zero_grad()
                loss.backward()

                optimizer.step()

            avg_loss = epoch_loss / batches
            history['train_loss'].append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.8f}")

        return history

    def predict(self, X):
        out = self.forward(Tensor(X))
        return out.data

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def save(self, path):
        # TODO: simpan state_dict (bobot & bias) ke file
        pass

    def load(self, path):
        # TODO: load bobot dan bias dari file ke model
        pass

    def plot_weights_distribution(self, layer_indices):
        # TODO: tampilkan grafik distribusi bobot dari layer-layer yang dipilih
        pass

    def plot_gradients_distribution(self, layer_indices):
        # TODO: tampilkan grafik distribusi gradien bobot dari layer-layer yang dipilih
        pass
