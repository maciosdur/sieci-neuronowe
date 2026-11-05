# Dla losowych wag porównujemy dL/dW z backward z ilorazem różnicowym
#   (L(w + eps) - L(w - eps)) / (2 eps)
import numpy as np
from lab3 import MLP, Linear, SigmoidBCELoss, X_train, y_train, n_features

EPS = 1e-6
N_CHECKS = 20

# Mała sieć z 2 warstwami ukrytymi (sprawdza też ReLU)
rng = np.random.default_rng(0)
model = MLP([n_features, 6, 5, 1], seed=0, std=0.5)
loss_fn = SigmoidBCELoss()
X, y = X_train[:30], y_train[:30]

loss_fn.forward(model.forward(X), y)
model.backward(loss_fn.backward())

print("Sprawdzenie gradientu: backward vs iloraz różnicowy")
worst = 0.0
for li, layer in enumerate(model.layers):
    if not isinstance(layer, Linear):
        continue
    for name, P, dP in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
        errs = []
        for _ in range(N_CHECKS):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + EPS
            l_plus = loss_fn.forward(model.forward(X), y)
            P[idx] = old - EPS
            l_minus = loss_fn.forward(model.forward(X), y)
            P[idx] = old
            num = (l_plus - l_minus) / (2 * EPS)
            errs.append(abs(num - dP[idx]) / max(abs(num) + abs(dP[idx]), 1e-12))
        worst = max(worst, max(errs))
        print(f"  warstwa Linear {li // 2 + 1}, {name}: max błąd względny = {max(errs):.2e}")
print(f"największy błąd względny: {worst:.2e}  ->  {'OK' if worst < 1e-5 else 'BŁĄD w backward'}")
