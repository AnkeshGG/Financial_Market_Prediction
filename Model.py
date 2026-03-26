# ==========================================
# FINAL PROJECT: SAE + KELM + SAVED GRAPHS
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy.linalg import inv


# ==========================================
# 1. CREATE RESULTS FOLDER
# ==========================================

def create_results_dir():
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_kaggle_data(file_path, symbol="SBIN"):
    df = pd.read_csv(file_path)

    if 'Symbol' in df.columns:
        df = df[df['Symbol'].str.upper() == symbol.upper()]
    elif 'SYMBOL' in df.columns:
        df = df[df['SYMBOL'].str.upper() == symbol.upper()]

    df = df.dropna().drop_duplicates()

    # Fix date format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')

    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype(float)

    return df[['Open','High','Low','Close']].values


# ==========================================
# 3. CREATE DATASET
# ==========================================

def create_dataset(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window][3])
    return np.array(X), np.array(y)


# ==========================================
# 4. AUTOENCODER
# ==========================================

class AutoEncoder:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.1

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        h = self.sigmoid(np.dot(x, self.W1))
        x_recon = self.sigmoid(np.dot(h, self.W2))
        return h, x_recon

    def train(self, X, lr=0.001, epochs=500):
        for _ in range(epochs):
            h, x_recon = self.forward(X)

            error = X - x_recon
            d_output = error * x_recon * (1 - x_recon)
            d_hidden = np.dot(d_output, self.W2.T) * h * (1 - h)

            self.W2 += lr * np.dot(h.T, d_output)
            self.W1 += lr * np.dot(X.T, d_hidden)

        return self.forward(X)[0]


class StackedAutoEncoder:
    def __init__(self, dims):
        self.layers = [AutoEncoder(dims[i], dims[i+1]) for i in range(len(dims)-1)]

    def train(self, X):
        for layer in self.layers:
            X = layer.train(X)
        return X

    def transform(self, X):
        for layer in self.layers:
            X, _ = layer.forward(X)
        return X


# ==========================================
# 5. KERNEL FUNCTIONS
# ==========================================

def polynomial_kernel(X1, X2, degree=2):
    return (1 + np.dot(X1, X2.T)) ** degree

def rbf_kernel(X1, X2, sigma=1.0):
    dist = np.sum(X1**2, axis=1).reshape(-1,1) + \
           np.sum(X2**2, axis=1) - 2*np.dot(X1, X2.T)
    return np.exp(-dist / (2 * sigma**2))


# ==========================================
# 6. KELM
# ==========================================

class KELM:
    def __init__(self, C=1.0, kernel='poly'):
        self.C = C
        self.kernel_type = kernel

    def kernel(self, X1, X2):
        return rbf_kernel(X1, X2) if self.kernel_type == 'rbf' else polynomial_kernel(X1, X2)

    def fit(self, X, y):
        self.X_train = X
        K = self.kernel(X, X)
        I = np.identity(K.shape[0])
        self.beta = np.dot(inv(K + I/self.C), y)

    def predict(self, X):
        K = self.kernel(X, self.X_train)
        return np.dot(K, self.beta)


# ==========================================
# 7. METRICS
# ==========================================

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ==========================================
# 8. GRAPH FUNCTIONS (SAVE)
# ==========================================

def plot_predictions(y_true, y_pred, folder):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.grid()
    plt.savefig(os.path.join(folder, "actual_vs_predicted.png"))
    plt.close()

def plot_error(y_true, y_pred, folder):
    plt.figure(figsize=(10,5))
    plt.plot(y_true - y_pred)
    plt.title("Error Plot")
    plt.grid()
    plt.savefig(os.path.join(folder, "error_plot.png"))
    plt.close()

def plot_trading_signal(y_true, y_pred, folder):
    signals = np.where(np.diff(y_pred) > 0, 1, -1)

    plt.figure(figsize=(10,5))
    plt.plot(y_true[:-1], label="Price")

    buy = np.where(signals == 1)[0]
    sell = np.where(signals == -1)[0]

    plt.scatter(buy, y_true[buy], marker='^', label="Buy")
    plt.scatter(sell, y_true[sell], marker='v', label="Sell")

    plt.legend()
    plt.title("Trading Signals")
    plt.savefig(os.path.join(folder, "trading_signals.png"))
    plt.close()


# ==========================================
# 9. PIPELINE
# ==========================================

def run_model(data):

    folder = create_results_dir()

    # Split
    split = int(0.7 * len(data))
    train_data = data[:split]
    test_data = data[split:]

    # Normalize
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Dataset
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Scale target
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1,1)).ravel()

    # SAE
    sae = StackedAutoEncoder([X_train.shape[1], 16, 8])
    X_train = sae.train(X_train)
    X_test = sae.transform(X_test)

    # KELM
    model = KELM(kernel='poly')  # or 'rbf'
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    print("MAPE:", MAPE(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", RMSE(y_test, y_pred))

    # Save graphs
    plot_predictions(y_test, y_pred, folder)
    plot_error(y_test, y_pred, folder)
    plot_trading_signal(y_test, y_pred, folder)

    print(f"\nGraphs saved in folder: {folder}/")

    return y_test, y_pred


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    data = load_kaggle_data("Nifty_200_scripts.csv", symbol="SBIN")
    run_model(data)