from .engine import Value
import numpy as np

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, nin, nout, activation='linear'):
        # TODO: inisialisasi bobot dan bias
        # TODO: implement metode inisialisasi: Zero, Uniform, Normal, Xavier, He
        pass

    def __call__(self, x):
        # TODO: operasi linear: out = (x @ W) + b
        # TODO: terapkan fungsi aktivasi (linear, relu, sigmoid, tanh, softmax)
        pass
    def parameters(self):
        # TODO: return list bobot dan bias
        return []

class FFNN(Module):
    def __init__(
        self, 
        layers_config,
        activations='relu', # bisa string atau array of strings
        loss_function='mse',
        init_method='normal',
        seed=42
    ):
        self.layers_config = layers_config
        self.loss_name = loss_function
        
        # TODO: Buat list self.layers berisi objek-objek Layer berdasarkan config
        # TODO: Tangani input activations baik berupa string tunggal atau list
        pass
    
    def fit(self, 
            X_train, 
            y_train, 
            X_val=None,
            y_val=None,
            batch_size=32,
            lr=0.01,
            epochs=100,
            verbose=1):
        
        history = {'train_loss': [], 'val_loss': []}

        # TODO: Implementasi Mini-batch SGD
        # TODO: Training loop per epoch:
        #   1. Forward pass
        #   2. Hitung Loss (MSE, BCE, atau CCE) & Regularisasi (L1/L2)
        #   3. Zero grad, lalu Backward pass
        #   4. Update bobot menggunakan optimizer
        #   5. Simpan histori loss & tangani verbose (progress bar)

        return history

    def predict(self, X):
        # TODO: Forward pass untuk prediksi akhir
        pass

    def forward(self, x):
        # TODO: Alirkan input x melalui setiap layer di self.layers
        pass

    def save(self, path):
        # Simpan state_dict (bobot dan bias) ke file
        pass

    def load(self, path):
        # TODO: Muat bobot dan bias dari file ke model
        pass

    def plot_weights_distribution(self, layer_indices):
        # TODO: Menampilkan grafik distribusi bobot dari layer-layer yang dipilih
        pass

    def plot_gradients_distribution(self, layer_indices):
        # TODO: Menampilkan grafik distribusi gradien bobot dari layer-layer yang dipilih
        pass
