import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

os.makedirs("wykresy", exist_ok=True)  

# ---------------------------------------------------------------
# 1. WCZYTANIE DANYCH
# ---------------------------------------------------------------
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features.copy()   
y = heart_disease.data.targets["num"]    

print("Rozmiar danych:", X.shape)

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# ---------------------------------------------------------------
# 2. CZY ZBIOR JEST ZBALANSOWANY?
# ---------------------------------------------------------------
print("\n--- Liczność klas (oryginalnie 0-4) ---")
print(y.value_counts().sort_index())

# licznosc klasy num
y.value_counts().sort_index().plot(kind="bar", title="Liczność klas przed binaryzacją (num = 0-4)")
plt.savefig("wykresy/klasy_0_4.png")
plt.close()

# binaryzajca 0 = brak choroby, 1 = choroba
y = (y > 0).astype(int)
print("\n--- Liczność klas po binaryzacji (0 = zdrowy, 1 = chory) ---")
print(y.value_counts().sort_index())

# wykres nowej klasy num
y.value_counts().sort_index().plot(kind="bar", title="Liczność klas (0 = zdrowy, 1 = chory)")
plt.savefig("wykresy/klasy.png")
plt.close()

# ---------------------------------------------------------------
# 3. SREDNIE I ODCHYLENIA CECH LICZBOWYCH
# ---------------------------------------------------------------
print("\n--- Średnia i odchylenie standardowe cech liczbowych ---")
print(X[num_cols].agg(["mean", "std"]).T.round(2))

# ---------------------------------------------------------------
# 4. CZy ROZKLAD NORMALNY?
#    Test Shapiro-Wilka: p > 0.05
# ---------------------------------------------------------------
print("\n--- Test normalności Shapiro-Wilka ---")
for col in num_cols:
    p = stats.shapiro(X[col]).pvalue
    print(f"{col:10s} p = {p:.4f} -> {'normalny' if p > 0.05 else 'NIE normalny'}")

# Histogramy cech liczbowych
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flatten(), num_cols):
    data = X[col]
    ax.hist(data, bins=20, density=True, alpha=0.6)
    xs = np.linspace(data.min(), data.max(), 200)
    ax.plot(xs, stats.gaussian_kde(data)(xs), color="black")
    ax.axvline(data.mean(), color="red", label=f"średnia = {data.mean():.1f}")
    ax.axvline(data.median(), color="green", linestyle="--", label=f"mediana = {data.median():.1f}")
    ax.set_title(f"{col}  (skośność = {stats.skew(data):.2f})")
    ax.legend(fontsize=8)
axes.flatten()[-1].axis("off")
plt.suptitle("Histogramy cech liczbowych")
plt.tight_layout()
plt.savefig("wykresy/histogramy.png")
plt.close()

# Boxploty cech liczbowych
fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 5))
for ax, col in zip(axes, num_cols):
    ax.boxplot(X[col])
    ax.set_title(col)
    ax.set_xticks([])
    ax.grid(axis="y")
plt.suptitle("Boxploty cech liczbowych")
plt.tight_layout()
plt.savefig("wykresy/boxploty.png")
plt.close()

# ---------------------------------------------------------------
# 5. RROZKLAD CECH KATEGORYCZNYCH
# ---------------------------------------------------------------
print("\n--- Test chi-square ---")
for col in cat_cols:
    counts = X[col].value_counts()
    p = stats.chisquare(counts).pvalue
    print(f"{col:8s} {counts.sort_index().to_dict()}  p = {p:.4f} -> {'równomierny' if p > 0.05 else 'NIE równomierny'}")

# Wykresy słupkowe cech kategorycznych
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.flatten(), cat_cols):
    X[col].value_counts().sort_index().plot(kind="bar", ax=ax, title=col)
plt.tight_layout()
plt.savefig("wykresy/kategoryczne.png")
plt.close()

# ---------------------------------------------------------------
# 6. BRAKI
# ---------------------------------------------------------------
print("\n--- Brakujące wartości ---")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Uzupelnienie dominatą
for col in ["ca", "thal"]:
    X[col] = X[col].fillna(X[col].mode()[0])

# ---------------------------------------------------------------
# 6 i pół MACIERZ KORELACJI CECH
# ---------------------------------------------------------------
corr = X.assign(target=y).corr()

plt.figure(figsize=(11, 9))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Macierz korelacji cech")
plt.tight_layout()
plt.savefig("wykresy/macierz_korelacji.png")
plt.close()

# ---------------------------------------------------------------
# 7. KODOWANIE I NORMALIZACJA
# ---------------------------------------------------------------
# One-hot encoding
X = pd.get_dummies(X, columns=cat_cols, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=67)

# Standaryzacja cech liczbowych
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\nLiczba cech po kodowaniu:", X_train.shape[1])

# ---------------------------------------------------------------
# 8. scikit learn
# ---------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Baseline: regresja logistyczna ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
print("Macierz pomyłek:\n", confusion_matrix(y_test, y_pred))

# Macierz pomyłek
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
plt.title("Macierz pomyłek - regresja logistyczna")
plt.savefig("wykresy/macierz_pomylek.png")
plt.close()

# ---------------------------------------------------------------
# 9. scikit learn dla targetu PRZED binaryzacja (klasy 0-4)
# ---------------------------------------------------------------
y5 = heart_disease.data.targets["num"]

X_train5, X_test5, y_train5, y_test5 = train_test_split(X, y5, test_size=0.2, stratify=y5, random_state=67)

# Standaryzacja cech liczbowych
scaler5 = StandardScaler()
X_train5[num_cols] = scaler5.fit_transform(X_train5[num_cols])
X_test5[num_cols] = scaler5.transform(X_test5[num_cols])

model5 = LogisticRegression(max_iter=1000)
model5.fit(X_train5, y_train5)
y_pred5 = model5.predict(X_test5)

print("\n--- Baseline: regresja logistyczna, klasy 0-4 ---")
print(f"Accuracy: {accuracy_score(y_test5, y_pred5):.3f}")
print(f"F1-score (macro): {f1_score(y_test5, y_pred5, average='macro'):.3f}")
print("Macierz pomyłek:\n", confusion_matrix(y_test5, y_pred5))

# Macierz pomyłek 5x5
ConfusionMatrixDisplay(confusion_matrix(y_test5, y_pred5)).plot()
plt.title("Macierz pomyłek - regresja logistyczna (klasy 0-4)")
plt.savefig("wykresy/macierz_pomylek_0_4.png")
plt.close()
