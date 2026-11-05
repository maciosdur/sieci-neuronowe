import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from lab3 import (MLP, Linear, SigmoidBCELoss, X_train, X_test, y_train, y_test, n_features,
                  metrics, mean_std)

SEEDS = [0, 1, 2]
PLATEAU_LOSS = 0.6   
TOL = 1e-6           
DIVERGE_FACTOR = 1e4 
HIDDEN = [n_features] * 5    
LR_GD = 0.01                 
LR_ADAM = 0.001              

warnings.filterwarnings("ignore", category=RuntimeWarning) 


# ---------------------------------------------------------------
# 1. INICJALIZACJA WAG
# ---------------------------------------------------------------
def linear_layers(model):
    return [layer for layer in model.layers if isinstance(layer, Linear)]


def init_weights(model, init, seed):
    rng = np.random.default_rng(seed)
    for layer in linear_layers(model):
        n_out, n_in = layer.W.shape
        std = np.sqrt(2 / n_in) if init == "he" else float(init)
        layer.W = rng.normal(0, std, (n_out, n_in))
        layer.b = np.zeros((1, n_out))


def build_model(hidden, init, seed):
    model = MLP([n_features] + hidden + [1], seed=seed)
    init_weights(model, init, seed)
    return model


# ---------------------------------------------------------------
# 2. OPTYMALIZATORY - step(model) aktualizuje W i b każdej warstwy Linear na podstawie dW, db z backward
# ---------------------------------------------------------------
class GD:
    def __init__(self, lr=LR_GD):
        self.lr = lr

    def step(self, model):
        for layer in linear_layers(model):
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db


class Adam:
    def __init__(self, lr=LR_ADAM, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m, self.v = {}, {}

    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, model):
        self.t += 1
        for i, layer in enumerate(linear_layers(model)):
            self.update((i, "W"), layer.W, layer.dW)
            self.update((i, "b"), layer.b, layer.db)


# ---------------------------------------------------------------
# 3. UCZENIE - batch_size: None = pełny zbiór, 1 = pojedyncze przykłady, k = paczki po k
# ---------------------------------------------------------------
def layer_stats(model, loss_fn):
    # forward + backward na CAŁYM zbiorze treningowym (bez kroku uczenia):
    # koszt, odchylenie wyjścia każdej warstwy Linear i norma pełnego gradientu |dW| każdej warstwy
    x, stds = X_train, []
    for layer in model.layers:
        x = layer.forward(x)
        if isinstance(layer, Linear):
            stds.append(x.std())
    loss = loss_fn.forward(x, y_train)
    model.backward(loss_fn.backward())
    return loss, stds, [np.linalg.norm(layer.dW) for layer in linear_layers(model)]


def train(hidden, init, optimizer, batch_size=None, epochs=1000, seed=0):
    model = build_model(hidden, init, seed)
    loss_fn = SigmoidBCELoss()
    rng = np.random.default_rng(seed + 1000)          # osobny generator do tasowania przykładów
    n = len(X_train)
    bs = n if batch_size is None else batch_size
    h = {"train_loss": [], "test_loss": [], "f1": [], "steps": [], "grad_norms": [], "diverged_at": None}
    steps = 0

    def evaluate():
        # indeks 0 = przed uczeniem, indeks k = po k epokach
        train_loss, stds, norms = layer_stats(model, loss_fn)
        test_loss = loss_fn.forward(model.forward(X_test), y_test)
        y_pred = (loss_fn.p.ravel() >= 0.5).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        h["train_loss"].append(train_loss)
        h["test_loss"].append(test_loss)
        h["f1"].append(2 * tp / (np.sum(y_pred) + np.sum(y_test)))
        h["steps"].append(steps)
        h["grad_norms"].append(norms)
        return stds

    with np.errstate(all="ignore"):                   # przy zepsutej inicjalizacji liczby wychodzą poza zakres
        h["init_stds"] = evaluate()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                loss_fn.forward(model.forward(X_train[idx]), y_train[idx])
                model.backward(loss_fn.backward())
                optimizer.step(model)
                steps += 1
            evaluate()
            # rozbieżność
            loss = h["train_loss"][-1]
            if not np.isfinite(loss) or loss > DIVERGE_FACTOR * max(h["train_loss"][0], 1.0):
                h["diverged_at"] = epoch
                break

    for k in ["train_loss", "test_loss", "f1", "steps"]:
        h[k] = np.array(h[k] + [np.nan] * (epochs + 1 - len(h[k])), dtype=float)
    norms = np.array(h["grad_norms"])
    h["grad_norms"] = np.vstack([norms, np.full((epochs + 1 - len(norms), norms.shape[1]), np.nan)])
    h["steps"] = np.arange(epochs + 1) * int(np.ceil(n / bs))
    return model, h


def summarize(model, h):
    with np.errstate(all="ignore"):
        m = metrics(model, X_test, y_test)
    tl = h["train_loss"]
    below = np.nonzero(tl < PLATEAU_LOSS)[0]
    m["plateau_end"] = below[0] if len(below) else np.nan
    m["diverged_at"] = h["diverged_at"] if h["diverged_at"] else np.nan
    m["train_loss"] = tl[~np.isnan(tl)][-1]
    m["test_loss"] = h["test_loss"][~np.isnan(h["test_loss"])][-1]
    m["min_test_loss"] = np.nanmin(h["test_loss"])
    # gdzie zatrzymałby się warunek stopu z lab2 (|zmiana kosztu| < TOL) i jaki byłby wtedy koszt
    small = np.nonzero(np.abs(np.diff(tl)) < TOL)[0]
    m["tol_stop_epoch"] = small[0] + 1 if len(small) else np.nan
    m["tol_stop_loss"] = tl[small[0] + 1] if len(small) else np.nan
    m["start_loss"] = tl[0]
    m["loss_e1"] = tl[1]
    m["max_loss"] = np.max(tl[np.isfinite(tl)])
    m["grad1_start"] = h["grad_norms"][0, 0]
    if h["diverged_at"]:
        for k in ["train_loss", "test_loss", "min_test_loss"]:
            m[k] = np.nan
    return m


# ---------------------------------------------------------------
# 4. EKSPERYMENT - każda konfiguracja dla 3 seedów, wydruk tabeli (średnia ± odchylenie)
# ---------------------------------------------------------------
def fmt(agg, k, p=".3g"):
    mean, std = agg[k]
    return "—" if np.isnan(mean) else f"{mean:{p}} ± {std:{p}}"


def experiment(configs, title, epochs):
    # configs: {nazwa: (hidden, init, funkcja tworząca optymalizator, batch_size)}
    print(f"\n=== {title} ===  ({epochs} epok, średnia ± odchylenie z {len(SEEDS)} seedów)")
    results = {}
    for name, (hidden, init, make_opt, bs) in configs.items():
        t = time.time()
        runs = [train(hidden, init, make_opt(), bs, epochs, seed) for seed in SEEDS]
        ms = [summarize(model, h) for model, h in runs]
        agg = {k: mean_std([m[k] for m in ms]) for k in ms[0]}
        n_stuck = sum(np.isnan(m["plateau_end"]) for m in ms)
        n_div = sum(not np.isnan(m["diverged_at"]) for m in ms)
        results[name] = (agg, [h for _, h in runs])
        print(f"{name:24s} koniec plateau: {fmt(agg, 'plateau_end', '.0f')} (nie wyszło: {n_stuck}/{len(SEEDS)})  "
              f"rozbieżność: {n_div}/{len(SEEDS)} (epoka {fmt(agg, 'diverged_at', '.0f')})  "
              f"stop |ΔL|<{TOL:g}: epoka {fmt(agg, 'tol_stop_epoch', '.0f')}, koszt {fmt(agg, 'tol_stop_loss')}  "
              f"[{(time.time() - t) / len(SEEDS):.1f} s / przebieg]")
        print(f"{'':24s} koszt train: start = {fmt(agg, 'start_loss')}  po 1. epoce = {fmt(agg, 'loss_e1')}  "
              f"maks. = {fmt(agg, 'max_loss')}  |dW| warstwy 1 na starcie = {fmt(agg, 'grad1_start')}")
        print(f"{'':24s} koszt train = {fmt(agg, 'train_loss')}  koszt test = {fmt(agg, 'test_loss')}  "
              f"min koszt test = {fmt(agg, 'min_test_loss')}  accuracy = {fmt(agg, 'accuracy', '.3f')}  "
              f"F1 = {fmt(agg, 'f1', '.3f')}  [F1 dla seedów: {', '.join(f'{m['f1']:.3f}' for m in ms)}]")
    return results


# ---------------------------------------------------------------
# 5. WYKRESY - każdy seed osobną cienką linią, ✕ = koniec przebiegu przerwanego przez rozbieżność
# ---------------------------------------------------------------
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LOSS_CLIP = (1e-6, 1e4)
SEEDS_NOTE = "cienkie linie = 3 seedy (każdy osobno), ✕ = przebieg przerwany z powodu rozbieżności"


def curve(h, key):
    return h["grad_norms"][:, 0] if key == "grad1" else h[key]


def draw(ax, results, names, colors, key, by_steps=False, clip=None):
    for name, color in zip(names, colors):
        for i, h in enumerate(results[name][1]):
            c = curve(h, key)
            x = h["steps"] if by_steps else np.arange(len(c))
            ok = np.isfinite(c)
            if not ok.any():
                continue
            y = np.clip(c, *clip) if clip else c
            ax.plot(x[ok], y[ok], color=color, lw=1.2, alpha=0.75, label=name if i == 0 else None)
            if h["diverged_at"]:
                last = np.nonzero(ok)[0][-1]
                ax.plot(x[last], y[last], "x", color=color, ms=10, mew=2.5)


def style(ax, title, ylabel, xlabel="epoka (skala symlog)", ylog=True, symlog=True, xlim=None):
    if symlog:
        ax.set_xscale("symlog", linthresh=1)
    if xlim:
        ax.set_xlim(*xlim)
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)


def save(fig, title, filename):
    fig.suptitle(f"{title}\n{SEEDS_NOTE}")
    fig.tight_layout()
    fig.savefig(f"wykresy/{filename}")
    plt.close(fig)


def plot_init(results):
    names = list(results)
    colors = COLORS[:len(names)]

    # (1) pierwsze 20 epok, wszystkie inicjalizacje - tu widać eksplozję N(0, 10) i N(0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    draw(axes[0], results, names, colors, "train_loss")
    style(axes[0], "(a) Koszt treningowy w pierwszych 20 epokach", "koszt train (skala log)",
          xlabel="epoka", symlog=False, xlim=(-0.5, 20))
    draw(axes[1], results, names, colors, "grad1")
    style(axes[1], "(b) Norma gradientu warstwy 1 w pierwszych 20 epokach", "|dW| warstwy 1 (skala log)",
          xlabel="epoka", symlog=False, xlim=(-0.5, 20))
    save(fig, "Inicjalizacja wag: początek uczenia (5 warstw ukrytych, GD lr = 0.01, pełny zbiór)",
         "cz2_inicjalizacja_start.png")

    # (2) całe uczenie bez N(0, 10) - ta rozbiega się w 1. epoce i rozciągnęłaby oś do 1e29
    keep = [n for n in names if "zepsuta" not in n]
    kcol = [colors[names.index(n)] for n in keep]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    draw(axes[0, 0], results, keep, kcol, "train_loss", clip=LOSS_CLIP)
    style(axes[0, 0], "(a) Koszt treningowy: czy i jak szybko spada", "koszt train (skala log)")
    draw(axes[0, 1], results, keep, kcol, "test_loss", clip=LOSS_CLIP)
    style(axes[0, 1], "(b) Koszt testowy: uogólnianie / przeuczenie", "koszt test (skala log)")
    draw(axes[1, 0], results, keep, kcol, "f1")
    style(axes[1, 0], "(c) F1 na zbiorze testowym", "F1 (test)", ylog=False)
    draw(axes[1, 1], results, keep, kcol, "grad1")
    style(axes[1, 1], "(d) Norma pełnego gradientu warstwy 1", "|dW| warstwy 1 (skala log)")
    save(fig, "Inicjalizacja wag: całe uczenie, 2000 epok (bez N(0, 10), która rozbiega się w 1. epoce)",
         "cz2_inicjalizacja.png")


def plot_gd_adam(results):
    rows = [("N(0, 0.01)", "GD, init 0.01", "ADAM, init 0.01"),
            ("He", "GD, init He", "ADAM, init He"),
            ("N(0, 1)", "GD, init 1", "ADAM, init 1")]
    cols = [("train_loss", "koszt treningowy", "koszt train (skala log)", LOSS_CLIP),
            ("test_loss", "koszt testowy", "koszt test (skala log)", LOSS_CLIP),
            ("grad1", "norma gradientu warstwy 1", "|dW| warstwy 1 (skala log)", None)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), sharey="col")
    for r, (label, gd, adam) in enumerate(rows):
        for c, (key, what, ylabel, clip) in enumerate(cols):
            ax = axes[r, c]
            draw(ax, results, [gd, adam], ["tab:blue", "tab:orange"], key, clip=clip)
            style(ax, f"init {label}: {what}", ylabel)
    save(fig, f"GD (lr = {LR_GD}) vs ADAM (lr = {LR_ADAM}), 5 warstw ukrytych, pełny zbiór, 2000 epok",
         "cz2_gd_vs_adam.png")


def plot_modes(results, label, filename):
    names = list(results)
    colors = COLORS[:len(names)]
    losses = np.concatenate([np.concatenate([h["train_loss"], h["test_loss"]]) for _, hs in results.values() for h in hs])
    losses = losses[np.isfinite(losses)]
    ylog = losses.max() / losses.min() > 1.5
    yl = "(skala log)" if ylog else "(skala liniowa)"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    draw(axes[0, 0], results, names, colors, "train_loss")
    style(axes[0, 0], "(a) Koszt treningowy względem EPOKI", f"koszt train {yl}", ylog=ylog)
    draw(axes[0, 1], results, names, colors, "train_loss", by_steps=True)
    style(axes[0, 1], "(b) Koszt treningowy względem LICZBY KROKÓW", f"koszt train {yl}",
          xlabel="liczba kroków uczenia (skala symlog)", ylog=ylog)
    draw(axes[1, 0], results, names, colors, "test_loss")
    style(axes[1, 0], "(c) Koszt testowy względem epoki", f"koszt test {yl}", ylog=ylog)
    draw(axes[1, 1], results, names, colors, "grad1")
    style(axes[1, 1], "(d) Norma pełnego gradientu warstwy 1", "|dW| warstwy 1 (skala log)")
    save(fig, f"Tryby uczenia, init {label} (5 warstw ukrytych, GD lr = {LR_GD}, 300 epok)", filename)


def plot_init_layers(results, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (agg, hs) in results.items():
        stds = np.mean([h["init_stds"] for h in hs], axis=0)
        norms = np.mean([h["grad_norms"][0] for h in hs], axis=0)
        layers = np.arange(1, len(stds) + 1)
        axes[0].plot(layers, stds, "o-", label=name)
        axes[1].plot(layers, norms, "o-", label=name)
    for ax, ylabel in zip(axes, ["odchylenie std wyjścia warstwy", "|dW| warstwy (pełny gradient)"]):
        ax.set_yscale("log")
        ax.set_xlabel("warstwa Linear (1 = wejściowa, 6 = wyjściowa)")
        ax.set_ylabel(ylabel)
        ax.grid(which="both", alpha=0.4)
        ax.legend()
    plt.suptitle(f"Stan sieci przed uczeniem (5 warstw ukrytych po {n_features}), średnia z {len(SEEDS)} seedów")
    plt.tight_layout()
    plt.savefig(f"wykresy/{filename}")
    plt.close()


# ---------------------------------------------------------------
# 6. TEST ADAM (szybki, sprawdza implementację na prostej funkcji)
# ---------------------------------------------------------------
def test_adam():
    # f(w) = (w - 3)^2, start w = 5. Dwie własności ADAM:
    #  - pierwszy krok ma długość lr niezależnie od wielkości gradientu (m_hat / sqrt(v_hat) = g / |g| = ±1)
    #  - po wielu krokach w -> 3
    print("=== Test ADAM na f(w) = (w - 3)^2, start w = 5, lr = 0.01 ===")
    for scale in [1.0, 1e6]:                          
        opt = Adam(lr=0.01)
        w = np.array([5.0])
        for t in range(1, 3001):
            opt.t = t
            opt.update("w", w, scale * 2 * (w - 3))
            if t == 1:
                first_step = 5.0 - w[0]
        print(f"skala gradientu {scale:6.0e}: pierwszy krok = {first_step:.5f}, w po 3000 krokach = {w[0]:.4f}")


# ---------------------------------------------------------------
# 7. EKSPERYMENTY
# ---------------------------------------------------------------
def main(scale=1.0):
    ep = lambda e: max(5, int(e * scale))
    t0 = time.time()
    test_adam()

    # EKSPERYMENT 1 - INICJALIZACJA (zwykły GD, pełny zbiór)
    res_init = experiment(
        {
            "init N(0, 0.01)": (HIDDEN, 0.01, GD, None),
            "init He": (HIDDEN, "he", GD, None),
            "init N(0, 1)": (HIDDEN, 1.0, GD, None),
            "init N(0, 10) (zepsuta)": (HIDDEN, 10.0, GD, None),
        },
        "Inicjalizacja wag (5 warstw ukrytych, GD lr = 0.01, pełny zbiór)", ep(2000),
    )
    plot_init(res_init)
    plot_init_layers(res_init, "cz2_inicjalizacja_warstwy.png")

    # EKSPERYMENT 2 - GD vs ADAM (pełny zbiór)
    res_adam = experiment(
        {
            "GD, init 0.01": (HIDDEN, 0.01, GD, None),
            "ADAM, init 0.01": (HIDDEN, 0.01, Adam, None),
            "GD, init He": (HIDDEN, "he", GD, None),
            "ADAM, init He": (HIDDEN, "he", Adam, None),
            "GD, init 1": (HIDDEN, 1.0, GD, None),
            "ADAM, init 1": (HIDDEN, 1.0, Adam, None),
        },
        f"GD (lr = {LR_GD}) vs ADAM (lr = {LR_ADAM}), 5 warstw ukrytych, pełny zbiór", ep(2000),
    )
    plot_gd_adam(res_adam)

    # EKSPERYMENT 3 - TRYBY UCZENIA (zwykły GD), osobno dla inicjalizacji He i 0.01
    for init, label, fname in [("he", "He", "cz2_tryby_he.png"), (0.01, "N(0, 0.01)", "cz2_tryby_001.png")]:
        res_modes = experiment(
            {
                "pełny zbiór (242)": (HIDDEN, init, GD, None),
                "paczki po 32": (HIDDEN, init, GD, 32),
                "pojedyncze przykłady (1)": (HIDDEN, init, GD, 1),
            },
            f"Tryby uczenia, init {label} (5 warstw ukrytych, GD lr = {LR_GD})", ep(300),
        )
        plot_modes(res_modes, label, fname)
    print(f"\nCałkowity czas: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
