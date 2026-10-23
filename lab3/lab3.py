import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("wykresy", exist_ok=True)

STD_INIT = 0.01    # wagi z N(0, 0.01) - z treści zadania
LR = 0.01          # współczynnik uczenia - z treści zadania
EPOCHS = 10000

# ---------------------------------------------------------------
# 1. DANE - przygotowanie tak samo jak w lab1/lab2
# ---------------------------------------------------------------
df = pd.read_csv("../lab1/heart_disease.csv")

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

y = (df["num"] > 0).astype(int).values           # 0 = zdrowy, 1 = chory
X = df.drop(columns="num")

for col in ["ca", "thal"]:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, columns=cat_cols, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=67)

# dane nieznormalizowane (do eksperymentu 2)
X_train_raw, X_test_raw = X_train.values.copy(), X_test.values.copy()

# dane znormalizowane
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
X_train, X_test = X_train.values, X_test.values

n_features = X_train.shape[1]
print("Train:", X_train.shape, " Test:", X_test.shape)


# ---------------------------------------------------------------
# 2. MODUŁY SIECI - każdy ma forward() i backward()
# ---------------------------------------------------------------
# X ma kształt (liczba próbek, liczba cech) - każdy wiersz to jeden pacjent.
# forward(x)  - liczy wyjście i zapamiętuje wejście x
# backward(d) - dostaje d = dL/dy (jak koszt zależy od wyjścia modułu),
#               zwraca dL/dx (jak koszt zależy od wejścia modułu)

class Linear:
    # y = W x + b
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, STD_INIT, (n_out, n_in))
        self.b = rng.normal(0, STD_INIT, (1, n_out))

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, d):
        self.dW = d.T @ self.x              # gradient po wagach - zapisujemy do kroku uczenia
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W                   # dL/dx


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d):
        return d * (self.x > 0)             # pochodna: 1 dla x > 0, 0 dla x <= 0


class Sigmoid:
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.y

    def backward(self, d):
        return d * self.y * (1 - self.y)    # pochodna sigmoidu: s(x) (1 - s(x))


class BCELoss:
    # koszt jak w regresji logistycznej (lab2)
    def forward(self, p, y):
        self.p = np.clip(p, 1e-12, 1 - 1e-12)
        self.y = y.reshape(-1, 1)
        return np.mean(-self.y * np.log(self.p) - (1 - self.y) * np.log(1 - self.p))

    def backward(self):
        return (self.p - self.y) / (self.p * (1 - self.p)) / len(self.y)   # dL/dp


class MLP:
    # [Linear -> ReLU] x liczba warstw ukrytych -> Linear -> Sigmoid
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.layers.append(Sigmoid())

    def forward(self, x):
        for layer in self.layers:              # od pierwszej warstwy do ostatniej
            x = layer.forward(x)
        return x

    def backward(self, d):
        for layer in reversed(self.layers):    # od ostatniej warstwy do pierwszej
            d = layer.backward(d)

    def step(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


def metrics(model, X, y):
    y_pred = (model.forward(X).ravel() >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------
# 3. UCZENIE
# ---------------------------------------------------------------
def train(hidden, X_tr, X_te, epochs=EPOCHS, lr=LR):
    # hidden - lista rozmiarów warstw ukrytych, np. [28] albo [28, 28]
    model = MLP([n_features] + hidden + [1])
    loss_fn = BCELoss()
    history = {"train_loss": [], "test_loss": [], "f1": []}

    for epoch in range(epochs):
        train_loss = loss_fn.forward(model.forward(X_tr), y_train)   # forward
        model.backward(loss_fn.backward())                            # backward
        model.step(lr)                                                # W = W - lr * dL/dW

        history["train_loss"].append(train_loss)
        history["test_loss"].append(loss_fn.forward(model.forward(X_te), y_test))
        y_pred = (loss_fn.p.ravel() >= 0.5).astype(int)    # predykcje z ostatniego forward (test)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        history["f1"].append(2 * tp / (np.sum(y_pred) + np.sum(y_test)))

    return model, history


def experiment(configs, title, filename):
    # configs: {nazwa: (warstwy ukryte, X_train, X_test)}
    print(f"\n=== {title} ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    results = {}
    for name, (hidden, X_tr, X_te) in configs.items():
        model, h = train(hidden, X_tr, X_te)
        m = metrics(model, X_te, y_test)
        results[name] = m
        print(f"{name:20s} koszt train = {h['train_loss'][-1]:.3f}  koszt test = {h['test_loss'][-1]:.3f}  "
              f"(min {min(h['test_loss']):.3f} w epoce {np.argmin(h['test_loss']) + 1:5d})  "
              f"accuracy = {m['accuracy']:.3f}  precision = {m['precision']:.3f}  "
              f"recall = {m['recall']:.3f}  F1 = {m['f1']:.3f}")
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["test_loss"], label=name)
        axes[2].plot(h["f1"], label=name)

    for ax, ylabel in zip(axes, ["koszt (train)", "koszt (test)", "F1 (test)"]):
        ax.set_xscale("log")
        ax.set_xlabel("epoka (skala log)")
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"wykresy/{filename}")
    plt.close()
    return results


# ---------------------------------------------------------------
# 4. EKSPERYMENT 1 - ROZMIAR WARSTWY UKRYTEJ
# ---------------------------------------------------------------
# h = liczba neuronów w warstwie ukrytej; liczba cech = 28
res_hidden = experiment(
    {f"h = {h}": ([h], X_train, X_test) for h in [2, 4, n_features, 128, 512]},
    f"Rozmiar warstwy ukrytej (1 warstwa ukryta, liczba cech = {n_features})",
    "rozmiar_warstwy.png",
)

# ---------------------------------------------------------------
# 5. EKSPERYMENT 2 - DANE ZNORMALIZOWANE I NIEZNORMALIZOWANE
# ---------------------------------------------------------------
res_norm = experiment(
    {
        "znormalizowane": ([n_features], X_train, X_test),
        "nieznormalizowane": ([n_features], X_train_raw, X_test_raw),
    },
    f"Normalizacja danych (1 warstwa ukryta, h = {n_features})",
    "normalizacja.png",
)

# ---------------------------------------------------------------
# 6. EKSPERYMENT 3 - LICZBA WARSTW
# ---------------------------------------------------------------
# 0 warstw ukrytych = sama regresja logistyczna
res_depth = experiment(
    {f"{d} warstw ukrytych": ([n_features] * d, X_train, X_test) for d in [0, 1, 2, 3]},
    f"Liczba warstw ukrytych (każda h = {n_features})",
    "liczba_warstw.png",
)
