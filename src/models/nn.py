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
        # Inisialisasi bobot dan bias
        # Pilihan: Zero, Uniform, Normal, Xavier, He
        pass

    def __call__(self, x):
        # Logika forward propagation per layer
        pass

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
        # ...
    
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

        # training loop (epoch & batch), simpan loss tiap epoch
        
        return history

    def predict(self, X):
        # Forward pass untuk prediksi akhir
        pass

    def forward(self, x):
        # Iterasi melalui semua layer
        pass
    
    def backward(self, x):
        # Memanggil proses backpropagation otomatis dari Value
        pass

    def save(self, path):
        # Menyimpan model
        pass

    def load(self, path):
        # Memuat model
        pass

    def plot_weights_distribution(self, layer_indices):
        # Menampilkan grafik distribusi bobot dari layer-layer yang dipilih
        pass

    def plot_gradients_distribution(self, layer_indices):
        # Menampilkan grafik distribusi gradien bobot dari layer-layer yang dipilih
        pass
    