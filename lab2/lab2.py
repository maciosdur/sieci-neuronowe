import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.makedirs("wykresy", exist_ok=True)

# ---------------------------------------------------------------
# 1. DANE - przygotowanie tak samo jak w lab1
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

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_train, X_test = X_train.values, X_test.values
n_features = X_train.shape[1]
print("Train:", X_train.shape, " Test:", X_test.shape)

# ---------------------------------------------------------------
# 2. MODEL - regresja logistyczna
# ---------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def predict_proba(X, W, b):
    return sigmoid(X @ W + b)


def loss(X, y, W, b):
    p = np.clip(predict_proba(X, W, b), 1e-12, 1 - 1e-12)
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))


def metrics(X, y, W, b):
    y_pred = (predict_proba(X, W, b) >= 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y == 1))   # chory dobrze trafiny
    fp = np.sum((y_pred == 1) & (y == 0))   # zdrowy za chorego
    fn = np.sum((y_pred == 0) & (y == 1))   # chory za zdrowego
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"accuracy": np.mean(y_pred == y), "precision": precision, "recall": recall, "f1": f1}


def metrics_sklearn(X, y, W, b):
    y_pred = (predict_proba(X, W, b) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


def init_weights(kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "zera":
        return np.zeros(n_features)
    if kind == "N(0, 0.01)":
        return rng.normal(0, 0.01, n_features)
    if kind == "N(0, 1)":
        return rng.normal(0, 1, n_features)
    if kind == "N(0, 10)":
        return rng.normal(0, 10, n_features)


def train(lr, init="zera", max_epochs=1000, tol=None, track=False):
    # tol - warunek stopu
    # track - zapis kosztu i metryk
    W = init_weights(init)
    b = 0.0
    history = {"train_loss": [], "test_loss": [], "f1": [], "precision": [], "recall": []}

    for epoch in range(max_epochs):
        p = predict_proba(X_train, W, b)

        # gradient
        grad_W = -X_train.T @ (y_train - p) / len(y_train)
        grad_b = -np.mean(y_train - p)

        W = W - lr * grad_W
        b = b - lr * grad_b

        history["train_loss"].append(loss(X_train, y_train, W, b))
        if track:
            history["test_loss"].append(loss(X_test, y_test, W, b))
            m = metrics(X_test, y_test, W, b)
            for k in ["f1", "precision", "recall"]:
                history[k].append(m[k])

        if tol is not None and epoch > 0 and abs(history["train_loss"][-2] - history["train_loss"][-1]) < tol:
            break

    return W, b, history


# ---------------------------------------------------------------
# 3. WPŁYW WSPÓŁCZYNNIKA UCZENIA
# ---------------------------------------------------------------
print("\n=== Wpływ współczynnika uczenia (inicjalizacja: zera, stop: zmiana kosztu < 1e-5) ===")
lrs = [0.001, 0.01, 0.1, 1, 10, 50]
histories_lr = {}
plt.figure(figsize=(9, 5))
for lr in lrs:
    W, b, h = train(lr, max_epochs=5000, tol=1e-5)
    histories_lr[lr] = h["train_loss"]
    if lr == 0.1:
        W_final, b_final = W, b
    m = metrics(X_test, y_test, W, b)
    print(f"lr = {lr:<6} epoki = {len(h['train_loss']):5d}  koszt train = {h['train_loss'][-1]:.4f}  "
          f"F1 = {m['f1']:.3f}  precision = {m['precision']:.3f}  recall = {m['recall']:.3f}")
    plt.plot(h["train_loss"], label=f"lr = {lr}")
plt.xscale("log")      
plt.yscale("log")      
plt.xlabel("epoka (skala log)")
plt.ylabel("koszt (train, skala log)")
plt.title("Krzywe uczenia dla różnych współczynników uczenia")
plt.legend()
plt.grid()
plt.savefig("wykresy/lr.png")
plt.close()



plt.figure(figsize=(9, 5))
for lr in lrs:
    epochs = range(1, len(histories_lr[lr][:100]) + 1)
    plt.plot(epochs, histories_lr[lr][:100], label=f"lr = {lr}", marker=".", markersize=3)
plt.xlabel("epoka")
plt.ylabel("koszt (train)")
plt.title("Pierwsze 100 epok - wszystkie lr (skala liniowa)")
plt.legend()
plt.grid()
plt.savefig("wykresy/lr_100_epok.png")
plt.close()

# ---------------------------------------------------------------
# 4. WPŁYW INICJALIZACJI WAG
# ---------------------------------------------------------------
print("\n=== Wpływ inicjalizacji (lr = 0.1, stop: zmiana kosztu < 1e-5) ===")
inits = ["zera", "N(0, 0.01)", "N(0, 1)", "N(0, 10)"]
plt.figure(figsize=(9, 5))
for init in inits:
    W, b, h = train(0.1, init=init, max_epochs=20000, tol=1e-5)
    m = metrics(X_test, y_test, W, b)
    print(f"{init:11s} epoki = {len(h['train_loss']):5d}  koszt początkowy = {h['train_loss'][0]:7.3f}  "
          f"koszt końcowy = {h['train_loss'][-1]:.4f}  F1 = {m['f1']:.3f}")
    plt.plot(h["train_loss"], label=init)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("epoka (skala log)")
plt.ylabel("koszt (train, skala log)")
plt.title("Krzywe uczenia dla różnych inicjalizacji wag (lr = 0.1)")
plt.legend()
plt.grid()
plt.savefig("wykresy/inicjalizacja.png")
plt.close()

# ---------------------------------------------------------------
# 5. LICZBA EPOK / WARUNEK STOPU - krzywa uczenia
# ---------------------------------------------------------------
W, b, h = train(0.1, max_epochs=3000, track=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(h["train_loss"], label="koszt train")
axes[0].plot(h["test_loss"], label="koszt test")
axes[0].set_title("Krzywa uczenia - koszt (lr = 0.1)")
axes[0].set_xlabel("epoka")
axes[0].legend()
axes[0].grid()

axes[1].plot(h["f1"], label="F1")
axes[1].plot(h["precision"], label="precision")
axes[1].plot(h["recall"], label="recall")
axes[1].set_title("Krzywa uczenia - metryki na zbiorze testowym")
axes[1].set_xlabel("epoka")
axes[1].legend()
axes[1].grid()


print("\n=== Warunek stopu (lr = 0.1) ===")
diffs = np.abs(np.diff(h["train_loss"]))   # zmiana kosztu miedzy epokami
for tol, color in [(1e-3, "red"), (1e-4, "orange"), (1e-5, "green")]:
    stop = np.argmax(diffs < tol) + 2     
    print(f"tol = {tol:.0e}  stop po {stop:4d} epokach  koszt test = {h['test_loss'][stop - 1]:.4f}  "
          f"F1 = {h['f1'][stop - 1]:.3f}")
    for ax in axes:
        ax.axvline(stop, color=color, linestyle="--", label=f"stop tol={tol:.0e}")
axes[0].legend()
plt.tight_layout()
plt.savefig("wykresy/krzywa_uczenia.png")
plt.close()

print(f"\nNajmniejszy koszt test w epoce {np.argmin(h['test_loss']) + 1}, "
      f"koszt test po 3000 epokach = {h['test_loss'][-1]:.4f} (min = {min(h['test_loss']):.4f})")

# ---------------------------------------------------------------
# 6. WYNIK KOŃCOWY
# ---------------------------------------------------------------
# model z sekcji 3 (lr = 0.1)
print("\n=== Model końcowy (lr = 0.1, zera, tol = 1e-5) ===")
m_np = metrics(X_test, y_test, W_final, b_final)
m_sk = metrics_sklearn(X_test, y_test, W_final, b_final)
print(f"{'metryka':10s} {'numpy':>7s} {'sklearn':>8s}")
for k in m_np:
    print(f"{k:10s} {m_np[k]:7.3f} {m_sk[k]:8.3f}")
