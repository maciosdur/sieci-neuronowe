import os
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STD_INIT = 0.01      # wagi z N(0, 0.01) 
LR = 0.01           
EPOCHS = 10000
SEEDS = [0, 1, 2]
PLATEAU_LOSS = 0.6

# ---------------------------------------------------------------
# 1. DANE - tak samo jak w lab1/lab2
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

# dane nieznormalizowane
X_train_raw, X_test_raw = X_train.values.copy(), X_test.values.copy()

# dane znormalizowane
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
X_train, X_test = X_train.values, X_test.values

n_features = X_train.shape[1]


# ---------------------------------------------------------------
# 2. MODUŁY SIECI - każdy ma forward() i backward()
# ---------------------------------------------------------------

class Linear:
    # y = W x + b
    def __init__(self, n_in, n_out, rng, std=STD_INIT):
        self.W = rng.normal(0, std, (n_out, n_in))
        self.b = rng.normal(0, std, (1, n_out))

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, d):
        self.dW = d.T @ self.x              
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W                   


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d):
        return d * (self.x > 0)             


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class SigmoidBCELoss:
    def forward(self, z, y):
        self.p = sigmoid(z)
        self.y = y.reshape(-1, 1)
        return np.mean(np.maximum(z, 0) - z * self.y + np.log1p(np.exp(-np.abs(z))))

    def backward(self):
        return (self.p - self.y) / len(self.y)


class MLP:
    # [Linear -> ReLU] x liczba warstw ukrytych -> Linear
    def __init__(self, sizes, seed=0, std=STD_INIT):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng, std))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                               

    def predict_proba(self, x):
        return sigmoid(self.forward(x)).ravel()

    def backward(self, d):
        for layer in reversed(self.layers): 
            d = layer.backward(d)

    def step(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db


def metrics(model, X, y):
    y_pred = (model.predict_proba(X) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------
# 3. UCZENIE
# ---------------------------------------------------------------
def train(hidden, X_tr, X_te, seed, epochs=EPOCHS, lr=LR):
    model = MLP([n_features] + hidden + [1], seed=seed)
    loss_fn = SigmoidBCELoss()
    history = {"train_loss": [], "test_loss": [], "f1": []}

    for epoch in range(epochs):
        train_loss = loss_fn.forward(model.forward(X_tr), y_train)   
        model.backward(loss_fn.backward())                            
        model.step(lr)                                                

        history["train_loss"].append(train_loss)
        history["test_loss"].append(loss_fn.forward(model.forward(X_te), y_test))
        y_pred = (loss_fn.p.ravel() >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        history["f1"].append(2 * tp / (np.sum(y_pred) + np.sum(y_test)))

    return model, {k: np.array(v) for k, v in history.items()}


def mean_std(values):
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def experiment(configs, title, filename):
    print(f"\n=== {title} ===  (średnia ± odchylenie z {len(SEEDS)} seedów)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_axis = np.arange(1, EPOCHS + 1)
    results = {}
    for name, (hidden, X_tr, X_te) in configs.items():
        runs = []
        for seed in SEEDS:
            model, h = train(hidden, X_tr, X_te, seed)
            m = metrics(model, X_te, y_test)
            m["train_loss"] = h["train_loss"][-1]
            m["test_loss"] = h["test_loss"][-1]
            m["min_test_loss"] = h["test_loss"].min()
            m["min_test_epoch"] = np.argmin(h["test_loss"]) + 1
            below = np.nonzero(h["train_loss"] < PLATEAU_LOSS)[0]
            m["plateau_end"] = below[0] + 1 if len(below) else np.nan
            runs.append((m, h))

        agg = {k: mean_std([m[k] for m, _ in runs]) for k in runs[0][0]}
        n_stuck = sum(np.isnan(m["plateau_end"]) for m, _ in runs)
        results[name] = agg

        def fmt(k):
            return f"{agg[k][0]:.3f} ± {agg[k][1]:.3f}"
        plateau = "—" if np.isnan(agg["plateau_end"][0]) else f"{agg['plateau_end'][0]:.0f} ± {agg['plateau_end'][1]:.0f}"
        print(f"{name:20s} koszt train = {fmt('train_loss')}  koszt test = {fmt('test_loss')}  "
              f"min koszt test = {fmt('min_test_loss')} (epoka {agg['min_test_epoch'][0]:.0f} ± {agg['min_test_epoch'][1]:.0f})  "
              f"koniec plateau: epoka {plateau} (nie wyszło: {n_stuck}/{len(SEEDS)})")
        print(f"{'':20s} accuracy = {fmt('accuracy')}  precision = {fmt('precision')}  "
              f"recall = {fmt('recall')}  F1 = {fmt('f1')}  "
              f"[F1 dla seedów: {', '.join(f'{m['f1']:.3f}' for m, _ in runs)}]")

        for ax, key in zip(axes, ["train_loss", "test_loss", "f1"]):
            curves = np.stack([h[key] for _, h in runs])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            line, = ax.plot(epochs_axis, mean, label=name)
            ax.fill_between(epochs_axis, mean - std, mean + std, color=line.get_color(), alpha=0.2)

    for ax, ylabel in zip(axes, ["koszt (train)", "koszt (test)", "F1 (test)"]):
        ax.set_xscale("log")
        ax.set_xlabel("epoka (skala log)")
        ax.set_ylabel(ylabel)
        ax.grid()
        ax.legend()
    plt.suptitle(f"{title}\nlinia = średnia z {len(SEEDS)} seedów, pas = ± odchylenie standardowe")
    plt.tight_layout()
    plt.savefig(f"wykresy/{filename}")
    plt.close()
    return results


def layers_label(d):
    if d == 1:
        return "1 warstwa ukryta"
    if d in (2, 3, 4):
        return f"{d} warstwy ukryte"
    return f"{d} warstw ukrytych"


if __name__ == "__main__":
    os.makedirs("wykresy", exist_ok=True)
    print("Train:", X_train.shape, " Test:", X_test.shape)

    # ---------------------------------------------------------------
    # 4. EKSPERYMENT 1 - ROZMIAR WARSTWY UKRYTEJ
    # ---------------------------------------------------------------
    # h = liczba neuronów w warstwie ukrytej
    experiment(
        {f"h = {h}": ([h], X_train, X_test) for h in [2, 4, n_features, 128, 512]},
        f"Rozmiar warstwy ukrytej (1 warstwa ukryta, liczba cech = {n_features})",
        "rozmiar_warstwy.png",
    )

    # ---------------------------------------------------------------
    # 5. EKSPERYMENT 2 - DANE ZNORMALIZOWANE I NIEZNORMALIZOWANE
    # ---------------------------------------------------------------
    experiment(
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
    experiment(
        {layers_label(d): ([n_features] * d, X_train, X_test) for d in [0, 1, 2, 3]},
        f"Liczba warstw ukrytych (każda h = {n_features})",
        "liczba_warstw.png",
    )
