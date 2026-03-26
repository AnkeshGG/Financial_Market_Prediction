# ==========================================
# SAE + KELM + PSO OPTIMIZER
# Paper: Mohanty et al., Applied Soft Computing 2021
# Optimizer added: Particle Swarm Optimization (PSO)
#   - Tunes KELM regularization parameter C
#   - Tunes RBF kernel sigma (or polynomial degree)
#   - Objective function: MAPE on validation split
# ==========================================

import cupy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from cupy.linalg import inv



# ==========================================
# 1. CREATE RESULTS FOLDER
# ==========================================

def create_results_dir():
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# ==========================================
# 2. LOAD DATA (unchanged)
# ==========================================

def load_kaggle_data(file_path, symbol="SBIN"):
    df = pd.read_csv(file_path)
    if 'Symbol' in df.columns:
        df = df[df['Symbol'].str.upper() == symbol.upper()]
    elif 'SYMBOL' in df.columns:
        df = df[df['SYMBOL'].str.upper() == symbol.upper()]
    df = df.dropna().drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(by='Date')
    df[['Open','High','Low','Close']] = df[['Open','High','Low','Close']].astype(float)
    return df[['Open','High','Low','Close']].values

# ==========================================
# 3. CREATE DATASET (unchanged)
# ==========================================

def create_dataset(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window][3])
    return np.array(X), np.array(y)

# ==========================================
# 4. AUTOENCODER (unchanged)
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
# 5. KERNEL FUNCTIONS (unchanged)
# ==========================================

def polynomial_kernel(X1, X2, degree=2):
    return (1 + np.dot(X1, X2.T)) ** degree

def rbf_kernel(X1, X2, sigma=1.0):
    dist = (np.sum(X1**2, axis=1).reshape(-1,1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T))
    return np.exp(-dist / (2 * sigma**2))

# ==========================================
# 6. KELM (modified: accepts sigma param)
# ==========================================

class KELM:
    def __init__(self, C=1.0, kernel='poly', sigma=1.0):
        self.C = C
        self.kernel_type = kernel
        self.sigma = sigma          # used by RBF kernel; PSO will tune this

    def kernel(self, X1, X2):
        if self.kernel_type == 'rbf':
            return rbf_kernel(X1, X2, sigma=self.sigma)
        else:
            return polynomial_kernel(X1, X2)

    def fit(self, X, y):
        self.X_train = X
        K = self.kernel(X, X)
        I = np.eye(K.shape[0])
        self.beta = np.linalg.solve(K + I / self.C, y)

    def predict(self, X):
        K = self.kernel(X, self.X_train)
        return np.dot(K, self.beta)

# ==========================================
# 7. METRICS (unchanged)
# ==========================================

def MAPE(y_true, y_pred):
    import numpy as _np
    return _np.mean(_np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def RMSE(y_true, y_pred):
    import numpy as _np
    return _np.sqrt(mean_squared_error(y_true, y_pred))

# ==========================================
# 8. PSO OPTIMIZER  ←  NEW
#
# What it does:
#   Each "particle" represents a candidate
#   (C, sigma) pair for KELM.
#   Particles fly through the search space,
#   attracted toward the best solution found
#   so far.  We minimise MAPE on a small
#   held-out validation split carved from
#   the training data.
#
# Parameters you can tune:
#   n_particles  – swarm size (20 is a good start)
#   n_iterations – how many rounds to run (30-50)
#   val_frac     – fraction of training data
#                  used for fitness evaluation
#   c_range      – log₂ search bounds for C
#   s_range      – search bounds for sigma
# ==========================================

class PSO_KELM_Optimizer:
    """
    Particle Swarm Optimizer for KELM hyperparameters.

    Search space (log scale for C, linear for sigma):
        C     ∈ [2^c_min, 2^c_max]
        sigma ∈ [s_min,   s_max  ]  (only used with RBF kernel)

    Fitness = MAPE on validation split (lower is better).
    """

    def __init__(self,
                 n_particles=20,
                 n_iterations=40,
                 val_frac=0.2,
                 c_range=(-5, 15),      # log2 bounds for C
                 s_range=(0.01, 10.0),  # bounds for sigma
                 kernel='rbf',
                 w=0.7,    # inertia weight
                 c1=1.5,   # cognitive (personal best) weight
                 c2=1.5,   # social (global best) weight
                 verbose=True):

        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.val_frac     = val_frac
        self.c_range      = c_range
        self.s_range      = s_range
        self.kernel       = kernel
        self.w            = w
        self.c1           = c1
        self.c2           = c2
        self.verbose      = verbose

        # Particle state: each row = [log2(C), sigma]
        self.positions  = None
        self.velocities = None
        self.p_best     = None   # personal best positions
        self.p_best_fit = None   # personal best fitness
        self.g_best     = None   # global best position
        self.g_best_fit = np.inf

        self.history = []        # g_best MAPE per iteration (for plotting)

    # --------------------------------------------------
    def _decode(self, position):
        """Convert raw particle position → (C, sigma)."""
        log2_C, sigma = position
        C     = 2 ** np.clip(log2_C, self.c_range[0], self.c_range[1])
        sigma = np.clip(sigma, self.s_range[0], self.s_range[1])
        return C, sigma

    # --------------------------------------------------
    def _fitness(self, position, X_tr, y_tr, X_val, y_val):
        """Train KELM with given (C, sigma) and return MAPE on val set."""
        C, sigma = self._decode(position)
        try:
            model = KELM(C=C, kernel=self.kernel, sigma=sigma)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            return MAPE(y_val, y_pred)
        except Exception:
            return np.inf   # invalid params (e.g. singular matrix)

    # --------------------------------------------------
    def optimize(self, X_train, y_train):
        """
        Run PSO.  Returns (best_C, best_sigma).

        X_train / y_train: SAE-encoded training features + targets.
        Internally splits off val_frac as a validation set for fitness.
        """
        # --- carve out validation split ---
        n_val  = max(1, int(len(X_train) * self.val_frac))
        X_tr   = X_train[:-n_val]
        y_tr   = y_train[:-n_val]
        X_val  = X_train[-n_val:]
        y_val  = y_train[-n_val:]

        # --- initialise swarm ---
        lo = np.array([self.c_range[0], self.s_range[0]])
        hi = np.array([self.c_range[1], self.s_range[1]])

        self.positions  = lo + np.random.rand(self.n_particles, 2) * (hi - lo)
        self.velocities = np.zeros_like(self.positions)
        self.p_best     = self.positions.copy()
        self.p_best_fit = np.array([
            self._fitness(p, X_tr, y_tr, X_val, y_val)
            for p in self.positions
        ])

        # initialise global best
        best_idx        = np.argmin(self.p_best_fit)
        self.g_best     = self.p_best[best_idx].copy()
        self.g_best_fit = self.p_best_fit[best_idx]

        # --- main PSO loop ---
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)

                # velocity update  (standard PSO formula)
                self.velocities[i] = (
                    self.w  * self.velocities[i]
                    + self.c1 * r1 * (self.p_best[i] - self.positions[i])
                    + self.c2 * r2 * (self.g_best    - self.positions[i])
                )

                # position update + clamp to search space
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i], lo, hi
                )

                # evaluate fitness
                fit = self._fitness(
                    self.positions[i], X_tr, y_tr, X_val, y_val
                )

                # update personal best
                if fit < self.p_best_fit[i]:
                    self.p_best[i]     = self.positions[i].copy()
                    self.p_best_fit[i] = fit

                # update global best
                if fit < self.g_best_fit:
                    self.g_best     = self.positions[i].copy()
                    self.g_best_fit = fit

            self.history.append(self.g_best_fit)

            if self.verbose:
                C_now, s_now = self._decode(self.g_best)
                print(f"  PSO iter {iteration+1:3d}/{self.n_iterations}"
                      f"  best MAPE={self.g_best_fit:.4f}%"
                      f"  C={C_now:.4f}  sigma={s_now:.4f}")

        best_C, best_sigma = self._decode(self.g_best)
        print(f"\n  PSO done. Best C={best_C:.4f}, sigma={best_sigma:.4f},"
              f" val MAPE={self.g_best_fit:.4f}%")
        return best_C, best_sigma

# ==========================================
# 9. GRAPH FUNCTIONS (one extra: PSO curve)
# ==========================================

def plot_predictions(y_true, y_pred, folder):
    plt.figure(figsize=(10,5))
    plt.plot(y_true,  label="Actual")
    plt.plot(y_pred,  label="Predicted")
    plt.legend(); plt.title("Actual vs Predicted"); plt.grid()
    plt.savefig(os.path.join(folder, "actual_vs_predicted.png"))
    plt.close()

def plot_error(y_true, y_pred, folder):
    plt.figure(figsize=(10,5))
    plt.plot(y_true - y_pred)
    plt.title("Error Plot"); plt.grid()
    plt.savefig(os.path.join(folder, "error_plot.png"))
    plt.close()

def plot_trading_signal(y_true, y_pred, folder):
    import numpy as _np   # force CPU numpy

    # ensure CPU arrays
    y_true = _np.asarray(y_true)
    y_pred = _np.asarray(y_pred)

    signals = _np.where(_np.diff(y_pred) > 0, 1, -1)

    plt.figure(figsize=(10,5))
    plt.plot(y_true[:-1], label="Price")

    buy  = _np.where(signals ==  1)[0]
    sell = _np.where(signals == -1)[0]

    plt.scatter(buy,  y_true[buy],  marker='^', label="Buy")
    plt.scatter(sell, y_true[sell], marker='v', label="Sell")

    plt.legend()
    plt.title("Trading Signals")
    plt.savefig(os.path.join(folder, "trading_signals.png"))
    plt.close()

def plot_pso_convergence(history, folder):
    # ensure CPU
    if hasattr(history, "get") or "cupy" in str(type(history)):
        history = np.asnumpy(np.array(history))

    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(history)+1), history, marker='o', markersize=3)
    plt.xlabel("PSO iteration")
    plt.ylabel("Best validation MAPE (%)")
    plt.title("PSO convergence curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "pso_convergence.png"))
    plt.close()

# ==========================================
# 10. PIPELINE (PSO inserted before KELM)
# ==========================================

def run_model(data):
    folder = create_results_dir()

    # --- split ---
    split      = int(0.7 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    # --- normalise (CPU - sklearn) ---
    scaler     = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data  = scaler.transform(test_data)

    # --- create dataset (CPU) ---
    X_train, y_train = create_dataset(train_data)
    X_test,  y_test  = create_dataset(test_data)

    X_train = np.asnumpy(X_train) if hasattr(X_train, "get") else X_train
    X_test  = np.asnumpy(X_test)  if hasattr(X_test, "get") else X_test

    # --- scale targets (CPU - sklearn) ---
    y_train = np.asnumpy(y_train) if hasattr(y_train, "get") else y_train
    y_test  = np.asnumpy(y_test)  if hasattr(y_test, "get") else y_test

    y_scaler = MinMaxScaler()
    y_train  = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()
    y_test   = y_scaler.transform(y_test.reshape(-1,1)).ravel()

    # ============================================
    # NOW move to GPU (after sklearn is done)
    # ============================================
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    # --- SAE ---
    print("Training SAE...")
    sae     = StackedAutoEncoder([X_train.shape[1], 16, 8])
    X_train = sae.train(X_train)
    X_test  = sae.transform(X_test)
    print("SAE done.")

    # --- PSO ---
    print("\nRunning PSO to optimise KELM hyperparameters...")
    pso = PSO_KELM_Optimizer(
        n_particles=20,
        n_iterations=40,
        val_frac=0.2,
        c_range=(-5, 15),
        s_range=(0.01, 10.0),
        kernel='rbf',
        w=0.7,
        c1=1.5,
        c2=1.5,
        verbose=True,
    )
    best_C, best_sigma = pso.optimize(X_train, y_train)

    plot_pso_convergence(np.asnumpy(np.array(pso.history)), folder)

    # --- final model ---
    print(f"\nRetraining KELM with optimised params: C={best_C:.4f}, sigma={best_sigma:.4f}")
    model = KELM(C=best_C, kernel='rbf', sigma=best_sigma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ============================================
    # Move back to CPU
    # ============================================
    y_test = np.asnumpy(y_test)
    y_pred = np.asnumpy(y_pred)

    # --- metrics ---
    y_test_cpu = y_test
    y_pred_cpu = y_pred

    print("\n===== Final Test Results =====")
    print(f"MAPE : {np.asnumpy(np.mean(np.abs((np.asarray(y_test_cpu) - np.asarray(y_pred_cpu)) / (np.asarray(y_test_cpu) + 1e-8)))) * 100:.4f} %")
    print(f"MAE  : {mean_absolute_error(y_test_cpu, y_pred_cpu):.6f}")
    print(f"RMSE : {RMSE(y_test_cpu, y_pred_cpu):.6f}")

    # --- plots ---
    plot_predictions(y_test, y_pred, folder)
    plot_error(y_test, y_pred, folder)
    plot_trading_signal(y_test, y_pred, folder)

    print(f"\nAll graphs saved in: {folder}/")
    return y_test, y_pred

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    data = load_kaggle_data("Nifty_200_scripts.csv", symbol="SBIN")
    run_model(data)