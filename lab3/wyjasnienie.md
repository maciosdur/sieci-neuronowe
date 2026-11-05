# Ćwiczenie 3 od zera: pełne wyjaśnienie kodu, matematyki i wniosków

Ten dokument tłumaczy **wszystko**, co jest w ćwiczeniu 3 (obie części): od tego, czym jest neuron, przez całą
matematykę propagacji wstecznej i ADAM, aż po każdą linijkę Twojego kodu i każdy wniosek z raportu.
Jest napisany tak, żeby dało się go czytać od zera, bez wcześniejszej wiedzy.

**Jak czytać:**
- rozdziały 1–7 to teoria części 1 (sieć, koszt, gradient, propagacja wsteczna);
- rozdział 8–9 to Twój kod z części 1 (`lab3.py`, `sprawdzenie_gradientu.py`) linijka po linijce;
- rozdział 10 to wyjaśnienie wyników i wniosków części 1;
- rozdziały 11–12 to teoria części 2 (optymalizacja, ADAM, inicjalizacja);
- rozdział 13 to kod części 2 (`lab3_cz2.py`) linijka po linijce;
- rozdział 14 to wyjaśnienie wyników części 2, łącznie z „dziwnymi” przypadkami;
- rozdział 15 to pytania, które może zadać prowadzący, z gotowymi odpowiedziami;
- rozdział 16 to słowniczek i ściąga na jedną stronę.

Wszystkie liczby w przykładach są policzone w Pythonie, nie „na oko”.

---

## Spis treści

1. [Zadanie i dane](#1-zadanie-i-dane)
2. [Pojedynczy neuron i regresja logistyczna](#2-pojedynczy-neuron-i-regresja-logistyczna)
3. [Funkcja kosztu: entropia krzyżowa](#3-funkcja-kosztu-entropia-krzyżowa)
4. [Gradient i spadek gradientu](#4-gradient-i-spadek-gradientu)
5. [Sieć wielowarstwowa (MLP) i ReLU](#5-sieć-wielowarstwowa-mlp-i-relu)
6. [Propagacja wsteczna: cała matematyka](#6-propagacja-wsteczna-cała-matematyka)
7. [Numeryczne sprawdzenie gradientu](#7-numeryczne-sprawdzenie-gradientu)
8. [Kod `lab3.py` linijka po linijce](#8-kod-lab3py-linijka-po-linijce)
9. [Kod `sprawdzenie_gradientu.py`](#9-kod-sprawdzenie_gradientupy)
10. [Wyniki części 1: co wyszło i dlaczego](#10-wyniki-części-1-co-wyszło-i-dlaczego)
11. [Teoria części 2: dlaczego uczenie głębokich sieci jest trudne](#11-teoria-części-2-dlaczego-uczenie-głębokich-sieci-jest-trudne)
12. [ADAM, tryby uczenia i inicjalizacja He: matematyka](#12-adam-tryby-uczenia-i-inicjalizacja-he-matematyka)
13. [Kod `lab3_cz2.py` linijka po linijce](#13-kod-lab3_cz2py-linijka-po-linijce)
14. [Wyniki części 2: co wyszło i dlaczego](#14-wyniki-części-2-co-wyszło-i-dlaczego)
15. [Pytania prowadzącego i odpowiedzi](#15-pytania-prowadzącego-i-odpowiedzi)
16. [Słowniczek i ściąga](#16-słowniczek-i-ściąga)
17. [Załącznik: pełne listingi kodu](#załącznik-pełne-listingi-kodu)

---

## 1. Zadanie i dane

### 1.1. Co w ogóle robimy?

Mamy pacjentów. Dla każdego znamy kilkanaście liczb (wiek, ciśnienie, cholesterol, wynik EKG…) i wiemy,
czy ma chorobę serca. Chcemy zbudować program, który **na podstawie tych liczb zgadnie, czy nowy pacjent jest chory**.

To jest **klasyfikacja binarna**: odpowiedź to jedna z dwóch klas, 0 = zdrowy, 1 = chory.

Program nie dostaje reguł („jeśli cholesterol > 240, to…”). Dostaje **przykłady** i sam ma znaleźć zależność.
To jest właśnie **uczenie maszynowe**, a sieć neuronowa to jeden ze sposobów, jak to zrobić.

### 1.2. Zbiór Heart Disease

Plik `../lab1/heart_disease.csv` ma **303 pacjentów** i 14 kolumn. Pierwsze trzy wiersze:

| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | num |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 63 | 1 | 1 | 145 | 233 | 1 | 2 | 150 | 0 | 2.3 | 3 | 0 | 6 | 0 |
| 67 | 1 | 4 | 160 | 286 | 0 | 2 | 108 | 1 | 1.5 | 2 | 3 | 3 | 2 |
| 67 | 1 | 4 | 120 | 229 | 0 | 2 | 129 | 1 | 2.6 | 2 | 2 | 7 | 1 |

Co znaczą kolumny:
- **liczbowe** (prawdziwe liczby, większa = więcej czegoś): `age` wiek, `trestbps` ciśnienie spoczynkowe,
  `chol` cholesterol, `thalach` maksymalne tętno, `oldpeak` obniżenie odcinka ST w EKG;
- **kategoryczne** (numer kategorii, a nie ilość): `sex` płeć, `cp` typ bólu w klatce (1–4), `fbs` cukier na czczo,
  `restecg` wynik EKG, `exang` ból przy wysiłku, `slope` nachylenie ST, `ca` liczba naczyń (0–3), `thal` wynik badania talu;
- **`num`**: stopień choroby 0–4. W oryginale jest 164 zdrowych (0) i 55 / 36 / 35 / 13 w stopniach 1–4.

### 1.3. Przygotowanie danych krok po kroku

**Krok 1: target binarny.** `y = (num > 0)`. Stopnie 1–4 to różne nasilenia tej samej choroby, więc łączymy je w „chory”.
Wychodzi 164 zdrowych i 139 chorych, czyli klasy są prawie zrównoważone.

**Krok 2: braki danych.** W kolumnie `ca` brakuje 4 wartości, w `thal` 2. Uzupełniamy je **dominantą** (najczęstszą wartością).
Przy 6 brakach na 303 pacjentów to najprostsze rozsądne rozwiązanie, które prawie nie zmienia danych.

**Krok 3: one-hot encoding cech kategorycznych.** Dlaczego? Weźmy `cp` (typ bólu) = 1, 2, 3, 4.
Gdybyśmy wpisali to do sieci jako liczbę, sieć „myślałaby”, że typ 4 to „dwa razy więcej” niż typ 2. To bez sensu,
bo to tylko numery kategorii (jak numery autobusów). One-hot zamienia jedną kolumnę na kilka kolumn 0/1:

| cp | cp_1 | cp_2 | cp_3 | cp_4 |
|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1 |

Teraz każda kategoria ma **własną wagę** w sieci i sieć może się nauczyć dowolnego wpływu każdej z nich.
Po zakodowaniu wszystkich 8 cech kategorycznych mamy **28 cech**: 5 liczbowych + 23 kolumny 0/1
(`sex_0, sex_1, cp_1…cp_4, fbs_0, fbs_1, restecg_0…2, exang_0, exang_1, slope_1…3, ca_0…ca_3, thal_3, thal_6, thal_7`).

**Krok 4: podział na zbiór treningowy i testowy.** `train_test_split(..., test_size=0.2, stratify=y, random_state=67)`:
- 80% (242 pacjentów) to **zbiór treningowy**: na nim sieć się uczy;
- 20% (61 pacjentów) to **zbiór testowy**: na nim sprawdzamy, czy sieć umie coś na **nowych** pacjentach;
- `stratify=y` pilnuje, żeby w obu zbiorach był ten sam odsetek chorych (około 46%). W teście jest 28 chorych i 33 zdrowych,
  w treningu 111 chorych na 242;
- `random_state=67` to stałe ziarno losowania: za każdym razem podział jest identyczny i taki sam jak w lab2,
  więc wyniki da się porównać z lab2.

Dlaczego test musi być osobno? Wyobraź sobie ucznia, który uczy się z tych samych zadań, które potem są na sprawdzianie.
Dostanie piątkę, ale nic to nie mówi o tym, czy umie przedmiot. Zbiór testowy to „sprawdzian z nowych zadań”.

**Krok 5: normalizacja (standaryzacja).** `StandardScaler` zamienia każdą cechę liczbową tak, żeby miała średnią 0
i odchylenie standardowe 1:

```
x_nowe = (x − średnia) / odchylenie
```

Dla Twoich danych (średnia i odchylenie liczone na zbiorze treningowym):

| Cecha | Średnia | Odchylenie | Przykład: wartość → po normalizacji |
|---|---|---|---|
| age | 54.4 | 8.9 | 63 → (63 − 54.4) / 8.9 = 0.97 |
| trestbps | 131.5 | 17.6 | 145 → 0.77 |
| chol | 247.7 | 51.5 | 233 → −0.29 |
| thalach | 149.7 | 22.5 | 150 → 0.01 |
| oldpeak | 1.0 | 1.2 | 2.3 → 1.08 |

Ważne szczegóły:
- `fit_transform` na treningu **liczy** średnią i odchylenie i od razu przekształca, a `transform` na teście **używa tych samych**
  liczb z treningu. Gdybyśmy liczyli średnią także z testu, „podglądalibyśmy” test (to się nazywa wyciek danych, *data leakage*);
- normalizujemy tylko 5 cech liczbowych. Cechy one-hot już są w skali 0/1;
- dlaczego normalizacja jest tak ważna, pokazuje eksperyment 2 (rozdział 10.3). W skrócie: bez niej cholesterol (około 250)
  jest setki razy większy niż cechy 0/1, a wtedy jeden współczynnik uczenia nie pasuje do wszystkich wag.

---

## 2. Pojedynczy neuron i regresja logistyczna

### 2.1. Neuron = ważona suma

Najprostszy „neuron” robi coś bardzo prostego: bierze cechy pacjenta, mnoży każdą przez jakąś liczbę (**wagę**),
dodaje wszystko i dodaje jeszcze jedną liczbę (**bias**, wyraz wolny):

```
z = w1·x1 + w2·x2 + … + w28·x28 + b
```

- **x** to cechy pacjenta (np. x1 = wiek po normalizacji);
- **w** to wagi: mówią, jak bardzo dana cecha „przemawia” za chorobą. Duża dodatnia waga = cecha zwiększa ryzyko,
  ujemna = zmniejsza, bliska zeru = nie ma znaczenia;
- **b** (bias) to „punkt startowy”, niezależny od cech.

Przykład z 2 cechami: w1 = 0.8 (wiek), w2 = −0.5 (maksymalne tętno), b = 0.1. Pacjent ma wiek 0.97 i tętno 0.01 (po normalizacji):

```
z = 0.8·0.97 + (−0.5)·0.01 + 0.1 = 0.776 − 0.005 + 0.1 = 0.871
```

Liczba **z** nazywa się **logitem**. Może być dowolna: −1000, 0, 5, 300. Nie jest jeszcze prawdopodobieństwem.

### 2.2. Sigmoid: zamiana logitu na prawdopodobieństwo

Chcemy liczby od 0 do 1 („jakie jest prawdopodobieństwo, że pacjent jest chory”). Używamy funkcji **sigmoid**:

```
σ(z) = 1 / (1 + e^(−z))
```

Jak się zachowuje:

| z | σ(z) | Znaczenie |
|---|---|---|
| −10 | 0.00005 | prawie na pewno zdrowy |
| −2 | 0.12 | raczej zdrowy |
| 0 | 0.5 | nie wiadomo |
| 0.871 | 0.70 | raczej chory |
| 2 | 0.88 | raczej chory |
| 10 | 0.99995 | prawie na pewno chory |

Sigmoid „ściska” całą oś liczbową do przedziału (0, 1). Dla dużych |z| jest prawie płaski (**nasycenie**),
co będzie ważne w części 2.

**Decyzja:** jeśli p = σ(z) ≥ 0.5, mówimy „chory” (1), w przeciwnym razie „zdrowy” (0). p ≥ 0.5 ⇔ z ≥ 0.

### 2.3. Regresja logistyczna = jeden neuron z sigmoidem

Neuron z sigmoidem to dokładnie **regresja logistyczna** z lab2. To ważne, bo:
- sieć bez warstw ukrytych (eksperyment 3, „0 warstw”) jest regresją logistyczną;
- dlatego wynik F1 = 0.836 dla 0 warstw zgadza się z lab2, co jest dowodem, że kod działa.

Ograniczenie regresji logistycznej: granica decyzji z = 0 to **płaszczyzna** (w 2D: prosta). Model umie tylko
„podzielić przestrzeń jedną prostą”. Jeśli chorzy i zdrowi są ułożeni w bardziej skomplikowany sposób, jeden neuron nie wystarczy.
Do tego są warstwy ukryte (rozdział 5).

---

## 3. Funkcja kosztu: entropia krzyżowa

### 3.1. Po co funkcja kosztu?

Sieć zgaduje prawdopodobieństwo p. Żeby się czegoś nauczyć, musi wiedzieć, **jak bardzo się pomyliła**.
Funkcja kosztu (*loss*) to jedna liczba: im większa, tym gorzej. Uczenie = szukanie takich wag, żeby koszt był jak najmniejszy.

### 3.2. Wzór

Dla jednego pacjenta (y = prawdziwa klasa, p = przewidziane prawdopodobieństwo choroby):

```
L = −[ y·ln(p) + (1 − y)·ln(1 − p) ]
```

Nie trzeba się bać tego wzoru, bo zawsze działa tylko jedna połowa:
- jeśli pacjent jest **chory** (y = 1): L = −ln(p). Chcemy, żeby p było blisko 1;
- jeśli **zdrowy** (y = 0): L = −ln(1 − p). Chcemy, żeby p było blisko 0.

Przykłady dla chorego pacjenta (y = 1):

| p (sieć mówi „chory z prawd.”) | Koszt −ln(p) | Komentarz |
|---|---|---|
| 0.99 | 0.01 | świetnie |
| 0.9 | 0.105 | dobrze |
| 0.5 | **0.693** | „nie wiem” |
| 0.1 | 2.30 | źle |
| 0.01 | 4.61 | bardzo źle |
| 10⁻¹⁰⁰ | 230 | katastrofa: pewna i błędna odpowiedź |

Dwie rzeczy z tej tabeli są kluczowe w całym ćwiczeniu:
1. **ln 2 ≈ 0.693 to koszt „zgadywania”.** Sieć z bardzo małymi wagami daje z ≈ 0, czyli p ≈ 0.5 dla każdego pacjenta,
   więc koszt wynosi 0.693. Dlatego „plateau na poziomie 0.69” na wykresach oznacza: sieć jeszcze nic nie umie.
2. **Pewna i błędna odpowiedź kosztuje ogromnie.** Koszt nie jest ograniczony z góry. To tłumaczy „dziwny” koszt testowy 75 w części 2.

Dla całego zbioru bierzemy **średnią** po wszystkich N pacjentach:

```
L = (1/N) · Σ_k  −[ y_k·ln(p_k) + (1 − y_k)·ln(1 − p_k) ]
```

### 3.3. Dlaczego logarytm, a nie np. (p − y)²?

- **Interpretacja:** to jest „ujemny logarytm prawdopodobieństwa, że model przewidział prawdziwą odpowiedź”.
  Minimalizowanie tego to tzw. metoda największej wiarygodności: szukamy modelu, przy którym prawdziwe dane są najbardziej prawdopodobne.
- **Dobry gradient:** w połączeniu z sigmoidem pochodna wychodzi po prostu **p − y** (rozdział 6.5). Przy kwadracie błędu
  pochodna byłaby (p − y)·p·(1 − p) i znikałaby, gdy sieć jest pewna i się myli (p ≈ 0 przy y = 1), czyli uczenie stawałoby
  dokładnie wtedy, gdy jest najbardziej potrzebne.
- Dla regresji logistycznej koszt jest **wypukły** (jedna „miska”, bez fałszywych dołków), co było tematem lab2.

### 3.4. Przykład: 0.6898, czyli „sieć zna tylko proporcję klas”

W części 2 zobaczysz koszt zatrzymujący się na 0.6898 zamiast 0.6931. Skąd? W zbiorze treningowym jest 111 chorych na 242,
czyli 45.9%. Jeśli sieć dla **każdego** pacjenta mówi p = 0.459 (nic nie wie o cechach, ale zna proporcję), to koszt:

```
L = −(0.459·ln 0.459 + 0.541·ln 0.541) = 0.6898
```

To minimalny koszt, jaki można osiągnąć **bez patrzenia na cechy**. Wszystko poniżej 0.6898 oznacza, że sieć faktycznie
korzysta z informacji o pacjencie.

---

## 4. Gradient i spadek gradientu

### 4.1. Intuicja: zejście z góry we mgle

Wyobraź sobie, że stoisz na zboczu góry we mgle i chcesz zejść na dół. Nie widzisz doliny, ale czujesz pod stopami,
w którą stronę teren opada. Robisz mały krok w dół, znowu sprawdzasz nachylenie, znowu krok… To jest **spadek gradientu**.

- „Wysokość” = koszt L;
- „położenie” = wartości wszystkich wag (u Ciebie od 29 do kilkunastu tysięcy liczb naraz);
- „nachylenie” = **gradient**, czyli wektor pochodnych ∂L/∂w dla każdej wagi.

### 4.2. Pochodna = nachylenie

Pochodna ∂L/∂w mówi: „jeśli zwiększę wagę w o malutką wartość, o ile zmieni się koszt?”.
- ∂L/∂w > 0: zwiększanie w zwiększa koszt, więc trzeba w **zmniejszyć**;
- ∂L/∂w < 0: trzeba w **zwiększyć**;
- ∂L/∂w ≈ 0: płasko, nie wiadomo, gdzie iść (to będzie problem plateau i punktów siodłowych).

Dlatego reguła uczenia ma minus:

```
w ← w − lr · ∂L/∂w
```

**lr** (*learning rate*, współczynnik uczenia) to długość kroku. U Ciebie lr = 0.01.

### 4.3. Przykład liczbowy: f(w) = (w − 3)²

Minimum jest w w = 3. Pochodna: f'(w) = 2(w − 3). Start w = 5, lr = 0.1:

| Krok | w | f'(w) = 2(w − 3) | Nowe w = w − 0.1·f'(w) |
|---|---|---|---|
| 1 | 5.000 | 4.000 | 4.600 |
| 2 | 4.600 | 3.200 | 4.280 |
| 3 | 4.280 | 2.560 | 4.024 |
| … | | | zbliża się do 3 |

Zauważ: im bliżej minimum, tym mniejszy gradient i tym mniejsze kroki. To naturalne hamowanie.

### 4.4. Za duży i za mały krok

- **Za mały lr** (np. 0.0001): poprawne, ale bardzo wolne. Potrzeba tysięcy kroków.
- **Za duży lr** (np. 1.1 w przykładzie wyżej): krok „przeskakuje” minimum na drugą stronę i ląduje **wyżej**
  niż był. w = 5 → 5 − 1.1·4 = 0.6 → 0.6 − 1.1·(−4.8) = 5.88 → … rozbiega się.

Ogólna zasada: lr musi być mniejszy niż około 2 / (krzywizna funkcji). Kierunek, w którym funkcja jest bardzo stroma
(duża krzywizna), wymaga małego lr. **To jest dokładnie problem danych nieznormalizowanych:** w kierunku wagi cholesterolu
funkcja jest tysiące razy bardziej stroma niż w kierunku cech 0/1, więc lr = 0.01 jest dla niej za duży (koszt skacze),
a dla cech 0/1 za mały.

### 4.5. Epoka, krok, paczka

- **Krok (iteracja)**: jedno policzenie gradientu i jedna aktualizacja wag.
- **Epoka**: jedno „przejrzenie” całego zbioru treningowego.
- W części 1 każda epoka to **jeden krok** liczony na wszystkich 242 pacjentach naraz (tzw. *full-batch gradient descent*).
- W części 2 dochodzą paczki (*mini-batch*, np. 32 pacjentów na krok, czyli 8 kroków na epokę) i pojedyncze przykłady
  (*SGD*, 242 kroki na epokę).

---

## 5. Sieć wielowarstwowa (MLP) i ReLU

### 5.1. Po co więcej warstw?

Jeden neuron rysuje jedną prostą (płaszczyznę). Klasyczny przykład, którego jedna prosta nie rozwiąże, to **XOR**:
punkty (0,0) i (1,1) to klasa 0, a (0,1) i (1,0) to klasa 1. Narysuj je na kartce: nie da się ich oddzielić jedną prostą.
Ale dwoma prostymi już tak. Warstwa ukryta to właśnie **kilka neuronów, z których każdy rysuje swoją prostą**,
a neuron wyjściowy łączy ich odpowiedzi w bardziej skomplikowaną granicę.

**MLP** (*multi-layer perceptron*) to sieć z warstwami:

```
wejście (28 cech) → [warstwa ukryta: h neuronów] → … → [neuron wyjściowy] → p
```

Każda warstwa robi dwie rzeczy:
1. **przekształcenie liniowe** `z = W·x + b` (każdy z h neuronów liczy swoją ważoną sumę);
2. **funkcję aktywacji** (u nas ReLU), która dodaje „zagięcie”.

### 5.2. Dlaczego aktywacja jest konieczna?

Bez aktywacji dwie warstwy liniowe to nadal jedna warstwa liniowa:

```
W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2) = W'·x + b'
```

Złożenie funkcji liniowych jest liniowe. Sto warstw bez aktywacji = jeden neuron. Aktywacja **nieliniowa** między warstwami
sprawia, że sieć może modelować krzywe granice.

### 5.3. ReLU

```
ReLU(z) = max(0, z)
```

Ujemne wartości zamienia na 0, dodatnie przepuszcza bez zmian. Przykład: ReLU([−2, 0.5, 3, −0.1]) = [0, 0.5, 3, 0].

Dlaczego ReLU, a nie np. sigmoid w warstwach ukrytych?
- **Pochodna jest prosta:** 1 dla z > 0, 0 dla z ≤ 0. Nie „ściska” gradientu jak sigmoid (pochodna sigmoidu to maksymalnie 0.25,
  więc przez 5 warstw gradient zmalałby co najmniej 4⁵ ≈ 1000 razy tylko przez aktywacje);
- **Szybka** do policzenia;
- standard w nowoczesnych sieciach i to, czego wymaga treść zadania.

Wada (ważna przy analizie wyników): ReLU **zeruje mniej więcej połowę** sygnału (wszystkie ujemne wartości). Stąd czynnik
„/2” we wzorach na wariancję i w inicjalizacji He.

### 5.4. Kształty macierzy w Twoim kodzie

W treści zadania (PDF) wektor wejściowy jest **kolumną** o kształcie (liczba cech, 1) i warstwa liczy `y = W·x`.
W Twoim kodzie dane są ułożone **wierszami**: macierz X ma kształt **(N, 28)**, każdy wiersz to jeden pacjent.
Dzięki temu liczymy całą paczkę pacjentów jednym mnożeniem macierzy:

```
Z = X · Wᵀ + b
(N, n_out) = (N, n_in) · (n_in, n_out) + (1, n_out)
```

- W ma kształt **(n_out, n_in)**, tak jak w PDF (wiersz = jeden neuron, kolumna = jedno wejście);
- transpozycja Wᵀ wynika z tego, że pacjenci są wierszami. To ta sama operacja co w PDF, tylko „położona na boku”:
  jeśli y = W·x dla kolumny x, to dla wiersza xᵀ: yᵀ = xᵀ·Wᵀ;
- b ma kształt (1, n_out) i numpy **automatycznie dodaje go do każdego wiersza** (tzw. *broadcasting*).

Przykład: sieć 28 → 28 → 1 na całym zbiorze treningowym:
- X: (242, 28), W1: (28, 28), Z1 = X·W1ᵀ + b1: (242, 28), A1 = ReLU(Z1): (242, 28);
- W2: (1, 28), Z2 = A1·W2ᵀ + b2: **(242, 1)**, czyli jeden logit na pacjenta;
- p = σ(Z2): (242, 1).

### 5.5. Ile wag ma sieć?

Warstwa n_in → n_out ma n_in·n_out wag + n_out biasów.
- 28 → 1 (regresja logistyczna): 29 parametrów;
- 28 → 28 → 1: 812 + 29 = 841;
- 28 → 512 → 1: 14 848 + 513 = 15 361;
- 5 warstw ukrytych po 28 (część 2): 5·812 + 29 = **4 089**, przy 242 pacjentach. To ważne przy przeuczeniu.

---

## 6. Propagacja wsteczna: cała matematyka

### 6.1. Problem

Mamy sieć z tysiącami wag i chcemy dla **każdej** wagi znać ∂L/∂w. Koszt zależy od wagi pierwszej warstwy pośrednio:
waga → wyjście warstwy 1 → wyjście warstwy 2 → … → p → L. Jak policzyć pochodną przez cały ten łańcuch?

### 6.2. Reguła łańcuchowa: przykład z życia

Jedziesz samochodem. Prędkość auta rośnie o 10 km/h na każdy cm wciśnięcia pedału. Zużycie paliwa rośnie o 0.1 l
na każde 1 km/h prędkości. O ile rośnie zużycie na każdy cm pedału? 10 · 0.1 = 1 l na cm. **Mnożymy „przełożenia”.**

Matematycznie: jeśli L zależy od y, a y od x, to:

```
∂L/∂x = ∂L/∂y · ∂y/∂x
```

Przy długim łańcuchu x → y1 → y2 → … → L:

```
∂L/∂x = ∂L/∂y_last · … · ∂y2/∂y1 · ∂y1/∂x
```

### 6.3. Pomysł propagacji wstecznej

Liczymy to od **końca**:
1. **forward**: przechodzimy przez sieć od wejścia do kosztu i **zapamiętujemy** wejście każdego modułu
   (będzie potrzebne do pochodnych);
2. **backward**: zaczynamy od ∂L/∂(wyjście sieci) i przechodzimy wstecz. Każdy moduł dostaje „ile koszt zależy od mojego wyjścia”
   (∂L/∂y), mnoży to przez swoją lokalną pochodną i oddaje „ile koszt zależy od mojego wejścia” (∂L/∂x) modułowi przed sobą.
   Moduł z wagami przy okazji liczy i zapisuje ∂L/∂W.

To dokładnie konwencja z PDF: `backward(dL/dy)` zwraca `dL/dx`.

**Dlaczego wstecz, a nie od przodu?** Bo koszt to **jedna liczba**, a wag jest tysiące. Idąc od końca, każdy moduł
liczy swoją pochodną raz i przekazuje dalej. Idąc od przodu, trzeba by śledzić wpływ każdej wagi osobno. Backprop kosztuje
mniej więcej tyle co jeden forward, niezależnie od liczby wag.

**Dlaczego forward musi zapamiętać wejścia?** Bo pochodne zależą od wartości z przejścia w przód. Np. ∂(W·x)/∂W = x:
żeby policzyć gradient wag, potrzebujemy x, które było na wejściu. Dlatego w kodzie jest `self.x = x`.

### 6.4. Moduł Linear: wyprowadzenie wzorów

Forward dla paczki (wiersze = pacjenci, k = numer pacjenta, j = numer neuronu, i = numer wejścia):

```
z[k, j] = Σ_i  x[k, i] · W[j, i]  +  b[j]
```

Backward dostaje `d[k, j] = ∂L/∂z[k, j]` (kształt (N, n_out)). Potrzebujemy trzech rzeczy.

**(a) Gradient po wagach ∂L/∂W[j, i].** Waga W[j, i] wpływa na z[k, j] dla **każdego** pacjenta k,
z pochodną ∂z[k, j]/∂W[j, i] = x[k, i]. Sumujemy wpływ po wszystkich pacjentach:

```
∂L/∂W[j, i] = Σ_k  d[k, j] · x[k, i]       →   dW = dᵀ · x
```

Kształty: dᵀ (n_out, N) · x (N, n_in) = **(n_out, n_in)**, czyli tyle, ile ma W. ✓

**(b) Gradient po biasie ∂L/∂b[j].** ∂z[k, j]/∂b[j] = 1, więc:

```
∂L/∂b[j] = Σ_k d[k, j]        →   db = suma d po wierszach (po pacjentach)
```

**(c) Gradient po wejściu ∂L/∂x[k, i]** (do przekazania warstwie wcześniej). Wejście x[k, i] wpływa na wszystkie neurony j
z pochodną W[j, i]:

```
∂L/∂x[k, i] = Σ_j  d[k, j] · W[j, i]       →   dx = d · W
```

Kształty: d (N, n_out) · W (n_out, n_in) = **(N, n_in)**, tyle ile ma x. ✓

Sztuczka do zapamiętania: **kształty muszą się zgadzać**. dW musi mieć kształt W, dx kształt x. Jest tylko jeden
sposób, żeby to wymnożyć, i to jest właściwy wzór.

Zauważ coś ważnego: **dW zawiera x**, a **dx zawiera W**. Jeśli wejścia x są malutkie, gradient wag jest malutki.
Jeśli wagi W są malutkie, gradient przekazany dalej wstecz jest malutki. To jest źródło zanikających gradientów (rozdział 10.4).

### 6.5. Moduł ReLU: dlaczego mnożenie element po elemencie

ReLU działa na każdy element osobno: y_i = max(0, x_i). Pełna macierz pochodnych ∂y/∂x ma na przekątnej
1 (gdy x_i > 0) albo 0 (gdy x_i ≤ 0), a poza przekątną same zera, bo y_i nie zależy od x_j dla j ≠ i.
Mnożenie wektora przez macierz diagonalną to po prostu mnożenie element po elemencie:

```
∂L/∂x = ∂L/∂y  ⊙  [x > 0]
```

To dokładnie uproszczenie, o którym mówi PDF („przemnożenie przez macierz diagonalną jest zastąpione mnożeniem element-wise”).
Intuicja: neuron, który w forward był „wyłączony” (dał 0), nie wpływa na koszt, więc gradient przez niego nie przechodzi.

(W punkcie x = 0 ReLU nie ma pochodnej. Przyjmujemy 0. W praktyce prawie nigdy nie trafiamy dokładnie w 0.)

### 6.6. Sigmoid + entropia krzyżowa: wyprowadzenie p − y

**Pochodna sigmoidu.** σ(z) = 1/(1 + e^(−z)):

```
σ'(z) = e^(−z) / (1 + e^(−z))²  =  [1/(1 + e^(−z))] · [e^(−z)/(1 + e^(−z))]  =  σ(z) · (1 − σ(z))
```

(Drugi nawias to 1 − σ(z), bo 1 − 1/(1+e^(−z)) = e^(−z)/(1+e^(−z)).)

**Pochodna kosztu po p.** L = −[y ln p + (1−y) ln(1−p)]:

```
∂L/∂p = −y/p + (1−y)/(1−p) = (p − y) / (p·(1 − p))
```

**Łańcuch:**

```
∂L/∂z = ∂L/∂p · ∂p/∂z = (p − y)/(p(1−p)) · p(1−p) = p − y
```

Piękny wynik: **gradient po logicie to po prostu błąd przewidywania**. Pacjent chory (y = 1), sieć mówi p = 0.3:
gradient −0.7, czyli „zwiększ logit”. Pacjent zdrowy, p = 0.3: gradient 0.3, „zmniejsz logit”.

Ponieważ koszt to **średnia** po N pacjentach, każdy pacjent wnosi 1/N:

```
∂L/∂z_k = (p_k − y_k) / N
```

Dlatego w kodzie jest `/ len(self.y)`. Dzięki temu gradient nie zależy od wielkości paczki (paczka 32 i 242 dają gradienty tej samej skali).

### 6.7. Dlaczego sigmoid i koszt połączone w jeden moduł (stabilność numeryczna)

Komputer liczy na liczbach zmiennoprzecinkowych float64, które mają około 16 cyfr znaczących. σ(40) = 1 − 4·10⁻¹⁸,
a to w float64 zaokrągla się do **dokładnie 1.0** (sprawdzone w Pythonie). Gdyby Sigmoid i koszt były osobnymi modułami:
- `Sigmoid.backward` mnożyłby przez p(1 − p) = 1·0 = **0**;
- gradient znika, choć sieć (z = 40, pacjent zdrowy) myli się maksymalnie i powinna dostać największą „karę”.

Połączony moduł liczy od razu p − y = 1 − 0 = **1**, bez dzielenia i mnożenia przez p(1 − p), które się skracają.

**Stabilny wzór na koszt.** Bezpośrednie ln(p) przy p = 0 daje −∞. Dlatego koszt liczymy z logitu. Wyprowadzenie:
ln σ(z) = −ln(1 + e^(−z)) oraz ln(1 − σ(z)) = −z − ln(1 + e^(−z)). Podstawiając:

```
L = −y·ln σ(z) − (1−y)·ln(1−σ(z)) = (1 − y)·z + ln(1 + e^(−z))
```

To działa dla z ≥ 0, ale dla dużego ujemnego z e^(−z) przepełnia się (e^1000 = ∞). Dla z < 0 przekształcamy
ln(1 + e^(−z)) = −z + ln(1 + e^z) i dostajemy L = −y·z + ln(1 + e^z). Oba przypadki łączy jeden wzór:

```
L = max(z, 0) − z·y + ln(1 + e^(−|z|))
```

Sprawdzenie: dla z ≥ 0: z − zy + ln(1 + e^(−z)) ✓; dla z < 0: 0 − zy + ln(1 + e^z) ✓. Wykładnik zawsze jest ≤ 0,
więc e^(…) jest między 0 a 1 i nigdy się nie przepełnia. `np.log1p(a)` liczy ln(1 + a) dokładnie także dla bardzo małych a.

### 6.8. Pełny przykład liczbowy: sieć 2 → 2 → 1, krok po kroku

Jeden pacjent z dwiema cechami x = [1, 2], chory (y = 1). Sieć: 2 wejścia → 2 neurony ukryte (ReLU) → 1 wyjście.

```
W1 = [[ 0.5, −0.2],      b1 = [0, 0]
      [ 0.1,  0.3]]
W2 = [[ 0.4, −0.6]]      b2 = [0.1]
```

**Forward:**

| Krok | Rachunek | Wynik |
|---|---|---|
| z1 = x·W1ᵀ + b1 | neuron 1: 0.5·1 + (−0.2)·2 = 0.1; neuron 2: 0.1·1 + 0.3·2 = 0.7 | [0.1, 0.7] |
| a1 = ReLU(z1) | oba dodatnie, bez zmian | [0.1, 0.7] |
| z2 = a1·W2ᵀ + b2 | 0.4·0.1 + (−0.6)·0.7 + 0.1 = 0.04 − 0.42 + 0.1 | −0.28 |
| p = σ(z2) | 1 / (1 + e^0.28) | 0.4305 |
| L = −ln p | | 0.8429 |

Sieć mówi „43% chory”, a pacjent jest chory, więc koszt jest wyższy niż 0.693 (gorzej niż zgadywanie).

**Backward** (od końca):

| Krok | Wzór | Wynik |
|---|---|---|
| dz2 = ∂L/∂z2 | p − y = 0.4305 − 1 | −0.5695 |
| dW2 = dz2ᵀ·a1 | −0.5695 · [0.1, 0.7] | [−0.0570, −0.3987] |
| db2 | dz2 | −0.5695 |
| da1 = dz2·W2 (przekazane do ReLU) | −0.5695 · [0.4, −0.6] | [−0.2278, 0.3417] |
| dz1 = da1 ⊙ [z1 > 0] | oba neurony aktywne, maska [1, 1] | [−0.2278, 0.3417] |
| dW1 = dz1ᵀ·x | [[−0.2278·1, −0.2278·2], [0.3417·1, 0.3417·2]] | [[−0.2278, −0.4556], [0.3417, 0.6835]] |
| db1 | dz1 | [−0.2278, 0.3417] |
| dx = dz1·W1 (gdyby była jeszcze warstwa wcześniej) | | [−0.0797, 0.1481] |

**Jak to czytać:**
- dW2[1] = −0.399 jest ujemne, więc trzeba **zwiększyć** W2[1] (z −0.6 w stronę zera). To zmniejszy „hamujący” wpływ
  neuronu 2, logit wzrośnie, p wzrośnie. Dokładnie tego chcemy, bo pacjent jest chory;
- dW2[1] jest 7 razy większe niż dW2[0], bo neuron 2 był 7 razy bardziej aktywny (0.7 vs 0.1). Waga przy mocniej
  świecącym neuronie ma większy wpływ na wynik, więc dostaje większą poprawkę.

**Krok uczenia** z lr = 0.1: W ← W − 0.1·dW dla wszystkich wag. Po kroku: **p = 0.494, L = 0.705** (było 0.431 i 0.843).
Jeden krok i sieć jest już bliżej prawdy.

**Sprawdzenie numeryczne** (rozdział 7): zmieniamy W2[0,0] o ±10⁻⁶ i liczymy (L₊ − L₋)/(2·10⁻⁶) = **−0.05695**,
dokładnie tyle samo co z backward.

---

## 7. Numeryczne sprawdzenie gradientu

### 7.1. Pomysł

Pochodna to z definicji granica ilorazu różnicowego. Możemy ją przybliżyć bez żadnych wzorów:

```
∂L/∂w ≈ ( L(w + ε) − L(w − ε) ) / (2ε),     ε = 10⁻⁶
```

Czyli: lekko zwiększ jedną wagę, policz koszt; lekko zmniejsz, policz koszt; podziel różnicę przez odległość.
To jest wolne (2 pełne forwardy na **jedną** wagę), ale bardzo proste i prawie nie da się tu popełnić błędu.
Dlatego używa się go jako **testu** dla szybkiego backwardu.

### 7.2. Dlaczego „centralny” iloraz (w + ε i w − ε)?

Iloraz jednostronny (L(w + ε) − L(w))/ε ma błąd rzędu ε (10⁻⁶). Iloraz centralny ma błąd rzędu ε² (10⁻¹²),
bo błędy z obu stron się znoszą (wynika to z rozwinięcia Taylora). Jest dużo dokładniejszy przy tym samym koszcie.

### 7.3. Błąd względny

Porównujemy liczbę z backwardu (a) i z ilorazu (b) wzorem:

```
błąd = |a − b| / (|a| + |b|)
```

Dzielimy przez wielkość liczb, bo różnica 0.001 jest ogromna dla gradientów rzędu 0.001, ale pomijalna dla gradientów rzędu 1000.
Interpretacja:
- około 10⁻⁷ lub mniej: backward poprawny (resztę błędu robi sam iloraz i zaokrąglenia);
- około 10⁻³–10⁻²: coś podejrzanego (np. trafiliśmy w „zagięcie” ReLU);
- około 0.1–1: błąd w backward.

U Ciebie największy błąd to **1.5·10⁻⁸**, więc backward jest poprawny we wszystkich trzech warstwach `Linear`
i w obu `ReLU`.

### 7.4. Dlaczego w teście wagi mają std = 0.5, a nie 0.01?

Przy wagach 0.01 gradienty głębszych warstw są rzędu 10⁻⁸ i mniejsze. Wtedy różnica L(w + ε) − L(w − ε) jest tak mała,
że ginie w zaokrągleniach float64 i błąd względny wychodzi duży, choć backward jest dobry. Większe wagi dają „zdrowe” gradienty
i test jest miarodajny. Poprawność wzorów nie zależy od wartości wag, więc test na std = 0.5 sprawdza ten sam kod.

---

## 8. Kod `lab3.py` linijka po linijce

Poniżej jest **cały** plik `lab3.py`, podzielony na kawałki. Pod każdym kawałkiem jest wyjaśnienie, co robi każda linijka i dlaczego.

### 8.1. Importy i ustawienie wątków

```python
import os
# macierze są małe (242 x 28 ... 512), więc wielowątkowe mnożenie macierzy tylko spowalnia
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))
```

- `os` – moduł do pracy z systemem (zmienne środowiskowe, katalogi).
- **Pętla `for var in [...]`**: numpy do mnożenia macierzy używa biblioteki BLAS (OpenBLAS albo MKL), która domyślnie dzieli
  każde mnożenie na wiele wątków (rdzeni procesora). Przy dużych macierzach to przyspiesza, ale nasze są małe (242 × 28),
  a mnożeń są setki tysięcy. Wtedy uruchamianie i synchronizacja wątków trwa dłużej niż samo liczenie.
  Ustawienie 1 wątku przyspieszyło cały skrypt z 270 s do 125 s (h = 512 nawet 6×), **bez zmiany wyników**.
- **Dlaczego przed `import numpy`?** Biblioteka BLAS czyta te zmienne tylko raz, w momencie ładowania numpy. Ustawione później nie działają.
- `numpy` – obliczenia na macierzach (cała sieć jest napisana w numpy).
- `pandas` – wczytanie CSV i przygotowanie tabeli (one-hot, braki).
- `matplotlib.pyplot` – wykresy.
- `sklearn` – tylko pomocniczo: podział danych, normalizacja i metryki końcowe. **Sama sieć i uczenie są napisane ręcznie**,
  bez sklearn i bez bibliotek do sieci neuronowych.
- `os.chdir(...)` – ustawia katalog roboczy na folder, w którym leży `lab3.py`. Dzięki temu ścieżki `../lab1/heart_disease.csv`
  i `wykresy/` działają niezależnie od tego, skąd uruchamiasz skrypt. `__file__` to ścieżka do bieżącego pliku.

### 8.2. Stałe

```python
STD_INIT = 0.01      # wagi z N(0, 0.01) - z treści zadania
LR = 0.01            # współczynnik uczenia - z treści zadania
EPOCHS = 10000
SEEDS = [0, 1, 2]    # każdą konfigurację uczymy od 3 różnych losowań wag początkowych
PLATEAU_LOSS = 0.6   # "wyjście z plateau" = pierwsza epoka z kosztem train < 0.6 (start: ln 2 ≈ 0.693)
```

- `STD_INIT = 0.01` – wagi losowane z rozkładu normalnego o średniej 0 i odchyleniu 0.01 (wymóg z treści).
- `LR = 0.01` – długość kroku w spadku gradientu (wymóg z treści).
- `EPOCHS = 10000` – tyle kroków uczenia. Dobrane tak, żeby sieci z 1–2 warstwami zdążyły wyjść z plateau (przy 1000 epok
  sieć z 2 warstwami w ogóle by nie ruszyła).
- `SEEDS` – trzy różne „ziarna” losowania wag. Ten sam seed = te same wagi początkowe = identyczny wynik przy każdym uruchomieniu.
- `PLATEAU_LOSS = 0.6` – próg, po którym uznajemy, że sieć „zaczęła się uczyć”. Start to 0.693 (zgadywanie), a 0.6 jest
  wyraźnie poniżej, ale osiągane szybko po wyjściu z plateau.

### 8.3. Wczytanie i przygotowanie danych

```python
df = pd.read_csv("../lab1/heart_disease.csv")

num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

y = (df["num"] > 0).astype(int).values           # 0 = zdrowy, 1 = chory
X = df.drop(columns="num")

for col in ["ca", "thal"]:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, columns=cat_cols, dtype=float)
```

- `pd.read_csv` – wczytuje tabelę 303 × 14 do obiektu `DataFrame`.
- `num_cols`, `cat_cols` – ręczny podział na cechy liczbowe i kategoryczne (rozdział 1.2).
- `y = (df["num"] > 0)` – porównanie daje kolumnę True/False, `.astype(int)` zamienia na 1/0, `.values` na tablicę numpy.
- `X = df.drop(columns="num")` – cechy to wszystko poza odpowiedzią (inaczej sieć „ściągałaby” odpowiedź z danych).
- `fillna(X[col].mode()[0])` – `mode()` zwraca najczęstsze wartości (może być kilka, stąd `[0]` = pierwsza),
  `fillna` wstawia ją w miejsce braków (NaN).
- `pd.get_dummies(..., dtype=float)` – one-hot encoding cech kategorycznych (rozdział 1.3). Kolumny liczbowe zostają bez zmian.
  Wynik: 28 kolumn.

```python
# podział stały (ten sam random_state co w lab2) - seedy zmieniają tylko wagi początkowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=67)

# dane nieznormalizowane (do eksperymentu 2)
X_train_raw, X_test_raw = X_train.values.copy(), X_test.values.copy()

# dane znormalizowane
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
X_train, X_test = X_train.values, X_test.values

n_features = X_train.shape[1]
```

- `train_test_split` – podział 242/61 ze stratyfikacją (rozdział 1.3).
- `X_train_raw = X_train.values.copy()` – **kopia przed normalizacją** do eksperymentu 2. `.copy()` jest konieczne:
  bez niego późniejsza normalizacja mogłaby zmienić też tę tablicę (współdzielona pamięć).
- `scaler.fit_transform` na treningu, `scaler.transform` na teście – średnia i odchylenie tylko z treningu (bez wycieku danych).
- `X_train[num_cols] = ...` – normalizowane są tylko kolumny liczbowe, one-hot zostaje 0/1.
- `.values` – zamiana z tabeli pandas na zwykłą macierz numpy (242, 28), bo sieć działa na numpy.
- `n_features = 28` – liczba kolumn, używana do budowy pierwszej warstwy.

**Ważne dla zrozumienia struktury:** cały ten kod jest **na poziomie modułu** (poza `if __name__ == "__main__"`), więc wykonuje się
także przy `import lab3`. Dzięki temu `sprawdzenie_gradientu.py` i `lab3_cz2.py` dostają gotowe `X_train`, `y_train` itd.
Trwa to ułamek sekundy, a eksperymenty (które są wolne) się nie uruchamiają.

### 8.4. Moduł `Linear`

```python
class Linear:
    # y = W x + b
    def __init__(self, n_in, n_out, rng, std=STD_INIT):
        self.W = rng.normal(0, std, (n_out, n_in))
        self.b = rng.normal(0, std, (1, n_out))

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, d):
        self.dW = d.T @ self.x              # gradient po wagach - zapisujemy do kroku uczenia
        self.db = d.sum(axis=0, keepdims=True)
        return d @ self.W                   # dL/dx
```

**`__init__`** (konstruktor, wywoływany przy tworzeniu warstwy):
- `rng.normal(0, std, (n_out, n_in))` – macierz wag z rozkładu normalnego N(0, std), kształt (liczba neuronów, liczba wejść).
  `rng` to generator liczb losowych przekazany z `MLP` (jeden na całą sieć, żeby kolejne warstwy dostały różne wagi,
  a całość była powtarzalna dla danego seeda);
- `self.b` – bias też losowany z N(0, 0.01), kształt (1, n_out), żeby dodawał się do każdego wiersza.
  (W części 2 bias jest zerowany, rozdział 13.2.)

Dlaczego wagi są **losowe**, a nie np. wszystkie równe 0? Gdyby wszystkie wagi były równe, wszystkie neurony w warstwie
liczyłyby to samo i dostawałyby identyczny gradient, więc zawsze pozostałyby identyczne. h neuronów działałoby jak 1.
Losowość „łamie symetrię”: każdy neuron startuje inaczej i uczy się czegoś innego.

**`forward(x)`**:
- `self.x = x` – zapamiętuje wejście, bo `backward` potrzebuje go do dW = dᵀ·x (rozdział 6.4);
- `x @ self.W.T + self.b` – `@` to mnożenie macierzy w Pythonie. (N, n_in)·(n_in, n_out) + (1, n_out) → (N, n_out).

**`backward(d)`** – d to ∂L/∂(wyjście), kształt (N, n_out):
- `self.dW = d.T @ self.x` – gradient wag, wzór (a) z rozdziału 6.4. **Zapisywany** w obiekcie, bo krok uczenia (`step`)
  wykona się dopiero po przejściu backward przez całą sieć;
- `self.db = d.sum(axis=0, keepdims=True)` – suma po pacjentach (oś 0 = wiersze), `keepdims=True` zostawia kształt (1, n_out)
  zamiast (n_out,), żeby pasował do `self.b`;
- `return d @ self.W` – ∂L/∂x, przekazywane do modułu wcześniej.

### 8.5. Moduł `ReLU`

```python
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, d):
        return d * (self.x > 0)             # pochodna: 1 dla x > 0, 0 dla x <= 0
```

- `np.maximum(0, x)` – element po elemencie max(0, x).
- `self.x > 0` – macierz True/False tego samego kształtu co x. W mnożeniu True = 1, False = 0, więc to jest „maska”.
- `d * (...)` – mnożenie element po elemencie (nie macierzowe!), rozdział 6.5. Gradient przechodzi tylko przez neurony,
  które w forward były aktywne.
- ReLU nie ma wag, więc nie liczy żadnego dW.

### 8.6. Funkcja `sigmoid`

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
```

- `np.clip(z, -500, 500)` – obcina z do przedziału [−500, 500]. Bez tego dla z = −1000 mielibyśmy e^1000 = ∞
  (przepełnienie i ostrzeżenie). σ(±500) jest i tak równe 0 albo 1 z dokładnością maszynową, więc obcięcie nie zmienia wyniku.
- Używana w `SigmoidBCELoss.forward` i w `MLP.predict_proba`.

### 8.7. Moduł `SigmoidBCELoss`

```python
class SigmoidBCELoss:
    # sigmoid + koszt jak w regresji logistycznej (lab2), połączone w jeden moduł.
    # Wejściem jest logit z (wyjście ostatniej warstwy Linear), a nie prawdopodobieństwo p.
    def forward(self, z, y):
        self.p = sigmoid(z)
        self.y = y.reshape(-1, 1)
        # -y ln p - (1-y) ln(1-p) zapisane stabilnie: max(z,0) - z y + ln(1 + e^-|z|)
        return np.mean(np.maximum(z, 0) - z * self.y + np.log1p(np.exp(-np.abs(z))))

    def backward(self):
        # dL/dz = dL/dp * dp/dz = (p - y) / (p(1-p)) * p(1-p) = p - y   (/ N, bo koszt to średnia)
        return (self.p - self.y) / len(self.y)
```

- **BCE** = *Binary Cross-Entropy*, czyli entropia krzyżowa dla dwóch klas (rozdział 3).
- `self.p = sigmoid(z)` – zapamiętuje prawdopodobieństwa do backward (i do liczenia F1 w pętli uczenia).
- `y.reshape(-1, 1)` – y przychodzi jako wektor (N,), a z ma kształt (N, 1). Bez reshape numpy przy odejmowaniu
  (N, 1) − (N,) zrobiłby macierz (N, N) (broadcasting) i wynik byłby kompletnie błędny. `-1` znaczy „ile trzeba”.
- `np.mean(...)` – stabilny wzór z rozdziału 6.7, uśredniony po pacjentach.
- `backward()` – nie dostaje argumentu, bo to ostatni moduł: ∂L/∂L = 1. Zwraca (p − y)/N (rozdział 6.6), kształt (N, 1).

### 8.8. Klasa `MLP`

```python
class MLP:
    # [Linear -> ReLU] x liczba warstw ukrytych -> Linear  (sigmoid jest w funkcji kosztu)
    def __init__(self, sizes, seed=0, std=STD_INIT):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng, std))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
```

- `sizes` – lista rozmiarów warstw, np. `[28, 28, 28, 1]` = wejście 28, dwie warstwy ukryte po 28, wyjście 1.
- `np.random.default_rng(seed)` – nowoczesny generator losowy numpy. Jeden na sieć, z ustalonym seedem.
- Pętla tworzy `Linear(sizes[i] → sizes[i+1])` dla każdej pary sąsiednich rozmiarów.
- `if i < len(sizes) - 2` – ReLU dodajemy **po każdej warstwie oprócz ostatniej**. Ostatnia Linear daje logit, a sigmoid jest w koszcie.
- Przykład: `sizes = [28, 28, 28, 1]` → `layers = [Linear(28→28), ReLU, Linear(28→28), ReLU, Linear(28→1)]`.
- `sizes = [28, 1]` → `[Linear(28→1)]` = regresja logistyczna.

```python
    def forward(self, x):
        for layer in self.layers:              # od pierwszej warstwy do ostatniej
            x = layer.forward(x)
        return x                               # logit z

    def predict_proba(self, x):
        return sigmoid(self.forward(x)).ravel()
```

- `forward` – wyjście jednej warstwy jest wejściem następnej. Każdy moduł po drodze zapamiętuje swoje wejście.
- `predict_proba` – prawdopodobieństwo choroby dla każdego pacjenta. `.ravel()` spłaszcza (N, 1) do (N,), bo tak chcą metryki sklearn.

```python
    def backward(self, d):
        for layer in reversed(self.layers):    # od ostatniej warstwy do pierwszej
            d = layer.backward(d)

    def step(self, lr):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db
```

- `backward` – `reversed` przechodzi od końca. Każdy moduł dostaje d od modułu „za sobą” i oddaje d dla modułu „przed sobą”.
  To jest cała propagacja wsteczna: **cztery linijki**, bo każdy moduł zna tylko swoją lokalną pochodną.
  Dzięki temu można zbudować sieć o dowolnej liczbie warstw bez wyprowadzania nowych wzorów (to pointa zadania z PDF).
- `step` – reguła uczenia w ← w − lr·∂L/∂w dla wszystkich warstw z wagami. `isinstance(layer, Linear)` pomija ReLU.
  `-=` zmienia tablicę „w miejscu”.

### 8.9. Funkcja `metrics`

```python
def metrics(model, X, y):
    y_pred = (model.predict_proba(X) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }
```

`y_pred` to decyzje 0/1 (próg 0.5). Żeby zrozumieć metryki, potrzebna jest **macierz pomyłek**. Przykład: h = 28, seed 0, zbiór testowy:

| | sieć mówi „chory” | sieć mówi „zdrowy” |
|---|---|---|
| **naprawdę chory (28)** | TP = 24 (trafiony chory) | FN = 4 (przeoczony chory) |
| **naprawdę zdrowy (33)** | FP = 6 (fałszywy alarm) | TN = 27 (trafiony zdrowy) |

- **accuracy** = (TP + TN) / wszyscy = 51/61 = **0.836**: jaki odsetek odpowiedzi jest poprawny;
- **precision** = TP / (TP + FP) = 24/30 = **0.800**: gdy sieć mówi „chory”, jak często ma rację;
- **recall** (czułość) = TP / (TP + FN) = 24/28 = **0.857**: jaki odsetek chorych sieć wykryła.
  W medycynie bardzo ważna, bo przeoczony chory jest gorszy niż fałszywy alarm;
- **F1** = 2·precision·recall / (precision + recall) = 2TP / (2TP + FP + FN) = 48/58 = **0.828**: średnia harmoniczna
  precision i recall. Jest niska, jeśli którakolwiek z nich jest niska.

`zero_division=0` – jeśli sieć **nikogo** nie uzna za chorego, precision = 0/0. Wtedy zamiast błędu wpisujemy 0.

Dwa przypadki skrajne, które pojawiają się w wynikach:
- sieć mówi „zdrowy” dla wszystkich: TP = 0, więc F1 = 0, a accuracy = 33/61 = **0.541** (odsetek zdrowych);
- sieć mówi „chory” dla wszystkich: TP = 28, FP = 33, F1 = 56/89 = **0.629**.

Metryki końcowe liczy sklearn, ale tylko **raz na przebieg**, po ostatniej epoce, więc na czas nie wpływa.

### 8.10. Funkcja `train`

```python
def train(hidden, X_tr, X_te, seed, epochs=EPOCHS, lr=LR):
    # hidden - lista rozmiarów warstw ukrytych, np. [28] albo [28, 28]
    model = MLP([n_features] + hidden + [1], seed=seed)
    loss_fn = SigmoidBCELoss()
    history = {"train_loss": [], "test_loss": [], "f1": []}
```

- `hidden` – tylko warstwy ukryte, np. `[]` (0 warstw), `[28]`, `[512]`, `[28, 28, 28]`.
  `[n_features] + hidden + [1]` dokleja wejście i wyjście, np. `[28] + [28, 28] + [1] = [28, 28, 28, 1]`.
- `X_tr, X_te` – osobno podawane dane, żeby ten sam `train` działał na danych znormalizowanych i surowych (eksperyment 2).
- `history` – słownik list, do których co epokę dopisujemy wyniki (do wykresów).

```python
    for epoch in range(epochs):
        train_loss = loss_fn.forward(model.forward(X_tr), y_train)   # forward
        model.backward(loss_fn.backward())                            # backward
        model.step(lr)                                                # W = W - lr * dL/dW
```

To jest **serce uczenia**, trzy linijki na epokę:
1. forward na **całym** zbiorze treningowym (242 pacjentów naraz) i koszt;
2. backward od kosztu do pierwszej warstwy (wypełnia `dW`, `db` w każdej `Linear`);
3. krok uczenia.

Szczegół: `train_loss` to koszt **przed** krokiem w tej epoce (forward był przed `step`).

```python
        history["train_loss"].append(train_loss)
        history["test_loss"].append(loss_fn.forward(model.forward(X_te), y_test))
        y_pred = (loss_fn.p.ravel() >= 0.5).astype(int)    # predykcje z ostatniego forward (test)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        history["f1"].append(2 * tp / (np.sum(y_pred) + np.sum(y_test)))

    return model, {k: np.array(v) for k, v in history.items()}
```

- Koszt testowy liczony po kroku. Ten forward na teście **nie** jest używany do uczenia (nie ma po nim backward ani step),
  tylko do obserwacji.
- `loss_fn.p` – po forward na teście trzyma prawdopodobieństwa dla 61 pacjentów testowych, więc nie trzeba liczyć ich drugi raz.
- **F1 liczone ręcznie w numpy** (szybkie, bo wywoływane 10 000 razy na przebieg):
  `(y_pred == 1) & (y_test == 1)` to „przewidziany chory i naprawdę chory”, czyli TP.
  `np.sum(y_pred)` = TP + FP, `np.sum(y_test)` = TP + FN = 28. Stąd 2TP / (2TP + FP + FN). Mianownik nigdy nie jest 0, bo w teście jest 28 chorych.
- Na końcu listy zamieniamy na tablice numpy (łatwiej liczyć średnie i rysować).

### 8.11. Funkcja `mean_std`

```python
def mean_std(values):
    # średnia i odchylenie z pominięciem nan (epoka wyjścia z plateau, gdy sieć nie wyszła)
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
```

- Liczy średnią i odchylenie standardowe z 3 seedów.
- `v[~np.isnan(v)]` – usuwa wartości NaN („brak”), np. „koniec plateau” dla seeda, który z plateau nie wyszedł.
  `~` to logiczne „nie”.
- Jeśli wszystkie są NaN (np. 3 warstwy: żaden seed nie wyszedł), zwraca (NaN, NaN), co w tabeli jest „—”.
- `v.std()` to odchylenie **populacyjne** (dzielenie przez n = 3, a nie n − 1 = 2). Przy 3 próbkach odchylenie z próby byłoby
  około 1.22 razy większe. Dobrze o tym wiedzieć, gdyby prowadzący zapytał (rozdział 15).

Funkcja jest też importowana przez `lab3_cz2.py`.

### 8.12. Funkcja `experiment`

```python
def experiment(configs, title, filename):
    # configs: {nazwa: (warstwy ukryte, X_train, X_test)}
    # każda konfiguracja uczona dla wszystkich SEEDS, wynik = średnia ± odchylenie standardowe
    print(f"\n=== {title} ===  (średnia ± odchylenie z {len(SEEDS)} seedów)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_axis = np.arange(1, EPOCHS + 1)   # od 1, bo na skali log punkt x = 0 by zniknął
    results = {}
```

- `configs` – słownik „nazwa na wykresie → (warstwy ukryte, dane treningowe, dane testowe)”.
- `plt.subplots(1, 3)` – jedna figura z trzema wykresami obok siebie (koszt train, koszt test, F1).
- `epochs_axis` od 1: na osi logarytmicznej log(0) nie istnieje, więc punkt x = 0 zniknąłby.

```python
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
```

Dla każdej konfiguracji i każdego seeda:
- uczy sieć (`train`) i liczy metryki końcowe;
- `[-1]` to ostatni element, czyli wartość po ostatniej epoce;
- `min_test_loss`, `min_test_epoch` – najmniejszy koszt testowy i kiedy wystąpił. `argmin` zwraca indeks od 0, stąd `+ 1`.
  To jest tylko informacja o przeuczeniu, **nie** wybieramy na jej podstawie modelu (to byłoby podglądanie testu);
- `np.nonzero(warunek)[0]` – indeksy epok, w których koszt < 0.6. Pierwszy z nich (`below[0]`) to koniec plateau.
  Jeśli lista jest pusta, sieć nie wyszła z plateau → NaN.

```python
        agg = {k: mean_std([m[k] for m, _ in runs]) for k in runs[0][0]}
        n_stuck = sum(np.isnan(m["plateau_end"]) for m, _ in runs)
        results[name] = agg
```

- `agg` – dla każdej metryki (klucza słownika) średnia ± odchylenie z 3 seedów. To jest tzw. *dict comprehension*:
  pętla, która od razu buduje słownik.
- `n_stuck` – ile seedów utknęło na plateau (True liczy się jako 1).

```python
        def fmt(k):
            return f"{agg[k][0]:.3f} ± {agg[k][1]:.3f}"
        plateau = "—" if np.isnan(agg["plateau_end"][0]) else f"{agg['plateau_end'][0]:.0f} ± {agg['plateau_end'][1]:.0f}"
        print(f"{name:20s} koszt train = {fmt('train_loss')}  koszt test = {fmt('test_loss')}  "
              f"min koszt test = {fmt('min_test_loss')} (epoka {agg['min_test_epoch'][0]:.0f} ± {agg['min_test_epoch'][1]:.0f})  "
              f"koniec plateau: epoka {plateau} (nie wyszło: {n_stuck}/{len(SEEDS)})")
        print(f"{'':20s} accuracy = {fmt('accuracy')}  precision = {fmt('precision')}  "
              f"recall = {fmt('recall')}  F1 = {fmt('f1')}  "
              f"[F1 dla seedów: {', '.join(f'{m['f1']:.3f}' for m, _ in runs)}]")
```

- `fmt` – mała funkcja formatująca „średnia ± odchylenie” z 3 miejscami po przecinku.
- `print(f"...")` – f-stringi: wyrażenia w `{}` są wstawiane do tekstu. `:20s` = wyrównanie do 20 znaków, `:.3f` = 3 miejsca po przecinku.
- Wypisuje dwie linijki na konfigurację (to, co jest w `wyniki.txt`), łącznie z F1 dla każdego seeda osobno.

```python
        for ax, key in zip(axes, ["train_loss", "test_loss", "f1"]):
            curves = np.stack([h[key] for _, h in runs])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            line, = ax.plot(epochs_axis, mean, label=name)
            ax.fill_between(epochs_axis, mean - std, mean + std, color=line.get_color(), alpha=0.2)
```

- `np.stack` – składa 3 krzywe (po jednej na seed) w macierz (3, 10000).
- `mean(axis=0)` – średnia po seedach w każdej epoce osobno.
- `ax.plot` rysuje linię średniej, `fill_between` rysuje półprzezroczysty pas od średnia − std do średnia + std
  w tym samym kolorze (`alpha=0.2` = 20% krycia). **Szeroki pas = wynik silnie zależy od losowania wag.**

```python
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
```

- **Oś X logarytmiczna**: równe odstępy dla 1, 10, 100, 1000, 10 000. Dzięki temu widać zarówno pierwsze epoki, jak i koniec uczenia.
  Na zwykłej osi pierwsze 100 epok zajmowałoby 1% szerokości wykresu.
- `tight_layout` – automatyczne odstępy, żeby podpisy się nie nakładały; `savefig` – zapis do pliku; `close` – zwolnienie pamięci.

### 8.13. Funkcja `layers_label`

```python
def layers_label(d):
    if d == 1:
        return "1 warstwa ukryta"
    if d in (2, 3, 4):
        return f"{d} warstwy ukryte"
    return f"{d} warstw ukrytych"
```

Tylko polska odmiana do legendy wykresu: 1 warstwa, 2–4 warstwy, 0 i 5+ warstw.

### 8.14. Część główna: eksperymenty

```python
if __name__ == "__main__":
    os.makedirs("wykresy", exist_ok=True)
    print("Train:", X_train.shape, " Test:", X_test.shape)

    experiment(
        {f"h = {h}": ([h], X_train, X_test) for h in [2, 4, n_features, 128, 512]},
        f"Rozmiar warstwy ukrytej (1 warstwa ukryta, liczba cech = {n_features})",
        "rozmiar_warstwy.png",
    )

    experiment(
        {
            "znormalizowane": ([n_features], X_train, X_test),
            "nieznormalizowane": ([n_features], X_train_raw, X_test_raw),
        },
        f"Normalizacja danych (1 warstwa ukryta, h = {n_features})",
        "normalizacja.png",
    )

    experiment(
        {layers_label(d): ([n_features] * d, X_train, X_test) for d in [0, 1, 2, 3]},
        f"Liczba warstw ukrytych (każda h = {n_features})",
        "liczba_warstw.png",
    )
```

- `if __name__ == "__main__":` – ten blok wykonuje się **tylko**, gdy uruchamiasz `python lab3.py`. Przy `import lab3`
  (z innych plików) Python ustawia `__name__ = "lab3"`, więc eksperymenty się nie uruchamiają. Dzięki temu część 2
  może korzystać z modułów części 1 bez czekania 4 minut.
- `os.makedirs("wykresy", exist_ok=True)` – tworzy folder, jeśli go nie ma (bez błędu, jeśli jest).
- **Eksperyment 1**: h ∈ {2, 4, 28, 128, 512}. Treść każe sprawdzić warstwę „równą liczbie cech, znacznie większą i znacznie mniejszą”:
  28 = równa, 128 i 512 = znacznie większe, 2 i 4 = znacznie mniejsze.
- **Eksperyment 2**: ta sama sieć (h = 28), dane znormalizowane vs surowe.
- **Eksperyment 3**: `[n_features] * d` to lista d razy po 28, np. d = 3 → `[28, 28, 28]`. d = 0 → `[]` (regresja logistyczna).

---

## 9. Kod `sprawdzenie_gradientu.py`

Cały plik:

```python
# Osobny, szybki test (< 1 s): czy backward() liczy pochodne poprawnie.
# Dla losowych wag porównujemy dL/dW z backward z ilorazem różnicowym
#   (L(w + eps) - L(w - eps)) / (2 eps)
# Moduły i dane bierzemy z lab3.py (import nie uruchamia eksperymentów).
import numpy as np
from lab3 import MLP, Linear, SigmoidBCELoss, X_train, y_train, n_features

EPS = 1e-6
N_CHECKS = 20     # ile losowych wag sprawdzamy w każdej macierzy W i wektorze b
```

- Importuje z `lab3.py` dokładnie te klasy, które są używane w eksperymentach. Test sprawdza **prawdziwy** kod, a nie kopię.
- `EPS = 1e-6` – wielkość zaburzenia ε. Za duże: iloraz źle przybliża pochodną. Za małe: różnica kosztów ginie w zaokrągleniach. 10⁻⁶ to standard.
- `N_CHECKS = 20` – nie sprawdzamy wszystkich wag (byłoby wolno), tylko 20 losowych w każdej macierzy W i wektorze b.

```python
# Mała sieć z 2 warstwami ukrytymi (sprawdza też ReLU). Wagi std = 0.5 zamiast 0.01,
# żeby gradienty nie były bliskie zeru - wtedy błąd względny jest miarodajny.
rng = np.random.default_rng(0)
model = MLP([n_features, 6, 5, 1], seed=0, std=0.5)
loss_fn = SigmoidBCELoss()
X, y = X_train[:30], y_train[:30]

loss_fn.forward(model.forward(X), y)
model.backward(loss_fn.backward())
```

- Sieć 28 → 6 → 5 → 1: dwie warstwy ukryte, więc test obejmuje Linear w trzech miejscach i ReLU dwa razy, w tym przekazywanie
  gradientu przez ReLU i przez Linear (`d @ W`). Małe rozmiary, bo szybciej.
- `std=0.5` – większe wagi, żeby gradienty były miarodajne (rozdział 7.4).
- 30 pierwszych pacjentów: wystarczy do testu, a forward jest szybszy.
- Jeden forward + backward wypełnia `dW` i `db` w każdej warstwie. To są wartości, które sprawdzamy.

```python
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
```

- `enumerate(model.layers)` – przechodzi po modułach z numerem (0 = Linear 1, 1 = ReLU, 2 = Linear 2, …). `continue` pomija ReLU.
- `li // 2 + 1` – zamienia indeks modułu na numer warstwy Linear (0 → 1, 2 → 2, 4 → 3). `//` to dzielenie całkowite.
- Pętla po `W` i `b`: `P` to parametr (macierz), `dP` to gradient z backward.
- `idx = tuple(rng.integers(0, s) for s in P.shape)` – losowa pozycja w macierzy, np. (3, 17).
- `old = P[idx]` – zapamiętuje oryginalną wartość. Potem: +ε, koszt; −ε, koszt; **przywrócenie** oryginału (ważne, żeby kolejne
  sprawdzenia działały na niezmienionej sieci).
- `P[idx] = ...` zmienia wagę **wewnątrz** modelu (P to ta sama tablica co `layer.W`, nie kopia), więc forward widzi zmianę.
- `num` – pochodna numeryczna; `errs` – błąd względny (rozdział 7.3). `max(..., 1e-12)` chroni przed dzieleniem przez 0, gdy obie pochodne są zerowe.
- Na końcu werdykt: OK, jeśli najgorszy błąd < 10⁻⁵.

Wynik (`wyniki_gradient.txt`): wszystkie błędy między 2.6·10⁻¹⁰ a 1.5·10⁻⁸, więc **OK**.

---

## 10. Wyniki części 1: co wyszło i dlaczego

### 10.1. Jak czytać wykresy z części 1

Każdy wykres ma 3 panele:
1. **koszt train** – jak dobrze sieć dopasowuje się do danych, na których się uczy (miara **optymalizacji**);
2. **koszt test** – jak dobrze radzi sobie na nowych pacjentach (miara **uogólniania**);
3. **F1 test** – jakość decyzji 0/1 na nowych pacjentach.

Linia to średnia z 3 seedów, pas to ± odchylenie standardowe. Oś X jest logarytmiczna (1, 10, 100, 1000, 10 000).

### 10.2. Najważniejsze zjawisko: plateau na początku uczenia

Na każdym wykresie (oprócz sieci bez warstw ukrytych) koszt przez setki epok stoi na **0.693** (zgadywanie),
a potem nagle gwałtownie spada. Dlaczego?

**Krok 1: wagi są malutkie.** Przy wagach 0.01 wyjście sieci jest prawie 0, więc p ≈ 0.5 dla każdego pacjenta i koszt = ln 2.

**Krok 2: gradient też jest malutki, i to „podwójnie”.** Weź sieć z 1 warstwą ukrytą: z = W2·ReLU(W1·x).
Z rozdziału 6.4:
- gradient wag **W2** jest proporcjonalny do wyjścia warstwy ukrytej, a ono jest małe, bo W1 jest małe;
- gradient wag **W1** jest proporcjonalny do W2 (gradient przechodzi wstecz przez W2), a W2 jest małe.

Każda warstwa „czeka” na drugą. Gdyby wszystkie wagi były **dokładnie** zerowe, gradient byłby dokładnie zerowy i sieć nigdy
by nie ruszyła. Punkt „wszystkie wagi = 0” to **punkt siodłowy** funkcji kosztu (więcej o nich w rozdziale 11). Inicjalizacja 0.01
startuje **tuż obok** tego punktu.

**Krok 3: dlaczego potem nagle spada?** Każdy krok trochę zwiększa wagi, a większe wagi dają większy gradient, który jeszcze
bardziej zwiększa wagi. To jest wzrost **wykładniczy**: przez długi czas wagi rosną z 0.01 do 0.02, 0.04… i nic nie widać,
a potem w kilkudziesięciu epokach osiągają rozsądną wielkość i koszt „spada z klifu”. Na osi logarytmicznej wygląda to
jak płaska półka i urwisko.

**Dlaczego sieć bez warstw ukrytych nie ma plateau?** Regresja logistyczna ma tylko jedną warstwę wag. Jej gradient
(p − y)·x **nie zależy** od wielkości wag. Nie ma tu „czekania na drugą warstwę”. Koszt spada od pierwszej epoki
(poniżej 0.6 po 48 epokach).

### 10.3. Eksperyment 1: rozmiar warstwy ukrytej

| h | Koniec plateau | Koszt train | Koszt test | Min. koszt test | F1 |
|---|---|---|---|---|---|
| 2 | 1882 ± 1094 | 0.296 | 0.371 | 0.350 | 0.816 ± 0.004 |
| 4 | 1360 ± 72 | 0.276 | 0.371 | 0.333 | 0.821 ± 0.000 |
| 28 | 907 ± 18 | 0.261 | 0.381 | 0.332 | 0.832 ± 0.007 |
| 128 | 609 ± 7 | 0.239 | 0.399 | 0.333 | 0.823 ± 0.007 |
| 512 | 402 ± 14 | 0.218 | 0.417 | 0.334 | 0.800 ± 0.010 |

**(a) Większe h szybciej wychodzi z plateau (1882 → 402 epok).** Neuron wyjściowy sumuje h składników W2_j·a_j.
Przy wagach losowych suma h niezależnych małych liczb rośnie jak √h: przy h = 512 sygnał jest około √(512/28) ≈ 4 razy
silniejszy niż przy h = 28. Silniejszy sygnał = większy gradient = krótsze „czekanie” na wykładniczy wzrost.

**(b) Przy h = 2 ogromne odchylenie (±1094 epok).** Przy dwóch neuronach wszystko zależy od tego, jak „szczęśliwie” wylosowano
2 kierunki. Jeden seed wyszedł z plateau dużo później niż pozostałe (szeroki niebieski pas na wykresie). Przy 28 i więcej
neuronach losowe różnice się uśredniają (prawo wielkich liczb), więc odchylenie jest małe (±7 do ±18).

**(c) Większe h = niższy koszt treningowy (0.296 → 0.218).** Więcej wag to większa „pojemność”: sieć może lepiej dopasować się
do konkretnych 242 pacjentów, także do ich przypadkowych cech (szumu).

**(d) Ale koszt testowy rośnie (0.371 → 0.417) – przeuczenie.** Najlepszy koszt testowy (minimum w trakcie uczenia) jest prawie
identyczny dla h ≥ 4 (około 0.333). Różnica jest w tym, co dzieje się **po** minimum: większa sieć szybciej i mocniej
„zapamiętuje” zbiór treningowy, więc na końcu (epoka 10 000) jest gorsza na teście.

**Przeuczenie (overfitting) na przykładzie:** uczeń zapamiętuje odpowiedzi do 242 konkretnych zadań zamiast zrozumieć metodę.
Na tych zadaniach ma coraz lepsze wyniki (koszt train spada), a na nowych coraz gorsze (koszt test rośnie).
Na wykresie: koszt train spada przez całe uczenie, koszt test ma dołek około epoki 2400–2900 i potem rośnie.

**(e) Czy różnica F1 jest istotna?** F1 0.832 (h = 28) vs 0.800 (h = 512) to około 2 pacjentów z 61. Dzięki 3 seedom widać,
że spadek dla h = 512 jest powtarzalny (0.793, 0.814, 0.793), więc to efekt przeuczenia, a nie przypadek. Jest jednak mały.
Nawet h = 2 daje F1 = 0.816.

### 10.4. Eksperyment 2: normalizacja

| Dane | Koszt train | Koszt test | F1 |
|---|---|---|---|
| znormalizowane | 0.261 ± 0.004 | 0.381 ± 0.005 | 0.832 ± 0.007 |
| surowe | 0.529 ± 0.013 | 0.692 ± 0.086 | 0.658 ± 0.019 |

**Dlaczego bez normalizacji jest tak źle?** Z rozdziału 6.4: gradient wagi pierwszej warstwy przy cesze i to
Σ_k d[k, j]·x[k, i], czyli jest **proporcjonalny do wartości cechy**.

Przykład: ten sam błąd d = 0.01 dla pacjenta z cholesterolem 250 i z cechą one-hot 1:
- gradient wagi „cholesterol”: 0.01 · 250 = **2.5**;
- gradient wagi „cp_4”: 0.01 · 1 = **0.01**.

Ten sam lr = 0.01 daje kroki 0.025 i 0.0001: **250 razy różne**. W kierunku cholesterolu funkcja kosztu jest bardzo
stroma (krzywizna rośnie jak kwadrat wartości cechy, 250² = 62 500 razy bardziej niż dla 0/1), więc lr = 0.01 jest za duży:
krok przeskakuje dolinę na drugą stronę i z powrotem (rozdział 4.4). Stąd **skoki kosztu** na wykresie.
A dla cech 0/1 lr jest za mały, więc te wagi prawie się nie uczą.

To tak, jakby dolina była wąskim, długim wąwozem: w poprzek ściany są strome (krok za duży, odbijamy się od ścian),
a wzdłuż prawie płasko (postęp minimalny).

**Ciekawostka z wykresu:** dane surowe wychodzą z plateau **wcześniej** (koszt zaczyna się ruszać już po kilkunastu epokach).
Dlaczego? Duże wartości wejść (np. 250) dają duży sygnał już w pierwszej warstwie mimo wag 0.01, więc nie ma problemu
„malutkiego sygnału”. Ale ruch jest chaotyczny: F1 skacze między 0.5 a 0.75 i nigdy nie dochodzi do 0.83.

**Duże odchylenie kosztu testowego (±0.086)** oznacza, że wynik zależy od seeda i od tego, w którym momencie oscylacji
zatrzymaliśmy uczenie. Niestabilny trening = niepowtarzalny wynik.

**Normalizacja** sprowadza wszystkie cechy do podobnej skali, „dolina” robi się okrągła i jeden lr pasuje do wszystkich wag.

### 10.5. Eksperyment 3: liczba warstw

| Warstwy ukryte | Koniec plateau | Koszt train | F1 |
|---|---|---|---|
| 0 | 48 ± 2 | 0.308 | 0.836 ± 0.000 |
| 1 | 907 ± 18 | 0.261 | 0.832 ± 0.007 |
| 2 | 9031 ± 575 | 0.386 ± 0.102 | 0.838 ± 0.003 |
| 3 | nie wyszła (0/3) | 0.690 | 0.000 |

**Każda warstwa wydłuża plateau około 10–20 razy.** Dlaczego tak drastycznie?

**Zanikanie sygnału w forward.** Każdy neuron liczy sumę 28 iloczynów waga × wejście. Z rachunku prawdopodobieństwa:
jeśli wagi są niezależne o odchyleniu σ, a wejścia mają średni kwadrat E[a²], to wariancja sumy n składników wynosi:

```
Var(z) = n · σ² · E[a²]
```

ReLU zeruje połowę wartości (te ujemne), więc E[a²] = ½·Var(z poprzedniej warstwy). Razem:

```
odchylenie(z) ≈ σ · √(n/2) · odchylenie(z poprzedniej warstwy) = 0.01 · √14 · … ≈ 0.037 · …
```

Czyli sygnał **słabnie około 27 razy na każdej warstwie**. Po 3 warstwach: 27³ ≈ 20 000 razy. Pomiar w części 2
(rozdział 14.2) dokładnie to potwierdza: 0.037, 0.0013, 5·10⁻⁵, …

**Zanikanie gradientu w backward.** Gradient przechodzi przez te same wagi (dx = d·W), więc też słabnie około 27 razy na warstwę.
Gradient pierwszej warstwy jest iloczynem wielu małych liczb.

**Dlaczego plateau rośnie tak szybko?** W sieci z L warstwami gradient każdej warstwy jest iloczynem wielkości **wszystkich
pozostałych** warstw. „Czekanie na wszystkie warstwy naraz” trwa tym dłużej, im więcej jest warstw: w przybliżeniu każda dodatkowa
warstwa mnoży czas wyjścia z plateau przez odwrotność skali sygnału na warstwę. Obserwowane: 48 → 907 → 9031 → (ponad 10 000).

**3 warstwy:** gradient tak mały, że przez 10 000 epok wagi praktycznie się nie zmieniły. Sieć mówi „zdrowy” dla wszystkich
(p minimalnie poniżej 0.5, bo zdrowych jest trochę więcej), stąd F1 = 0 i accuracy = 0.541.

**2 warstwy: ±0.1 w koszcie treningowym.** Sieć wyszła z plateau dopiero około epoki 9000, więc po 10 000 epokach jest
w środku „spadania z klifu”. Każdy seed jest w innym miejscu tego spadku, więc koszty bardzo się różnią.

### 10.6. Dlaczego MLP nie jest lepsze od regresji logistycznej?

Wszystkie sieci, które się nauczyły, osiągają F1 około 0.83–0.84, tyle co regresja logistyczna. Dlaczego warstwy ukryte nic nie dają?

- **Zbiór nie jest liniowo separowalny** (gdyby był, regresja logistyczna doszłaby do kosztu treningowego 0; zatrzymuje się na 0.31
  i myli się na około 15% pacjentów testowych);
- ale **zależność jest w przybliżeniu liniowa**: ryzyko rośnie mniej więcej proporcjonalnie do kilku cech (wiek, typ bólu, liczba naczyń…).
  Model liniowy wyciąga prawie całą informację, a reszta błędów to szum, którego żaden model nie przewidzi z tych danych;
- dodatkowe warstwy mogą modelować bardziej złożone zależności, ale tu takich nie ma, więc zamiast tego uczą się szumu (przeuczenie);
- 242 przykłady to bardzo mało dla sieci z setkami wag.

**Wniosek, który warto powiedzieć:** „Lepszy model to nie zawsze większy model. Złożoność modelu trzeba dopasować do ilości danych
i do złożoności zależności, a na Heart Disease prosta regresja logistyczna wystarcza.”

### 10.7. Dlaczego 3 seedy i co znaczy „±”

Wynik sieci zależy od losowych wag początkowych. Jeden przebieg może mieć „szczęście” albo „pecha”. Przy 61 pacjentach
testowych zmiana decyzji dla jednego pacjenta zmienia F1 o około 0.015–0.02. Bez powtórzeń nie da się odróżnić
**prawdziwego efektu** (np. h = 512 się przeucza) od **przypadku**.

- „0.832 ± 0.007” znaczy: średnia z 3 przebiegów to 0.832, a typowe odchylenie pojedynczego przebiegu od średniej to 0.007.
- Małe ± (np. h = 28 koniec plateau 907 ± 18): wynik powtarzalny, nie zależy od losowania.
- Duże ± (np. h = 2: 1882 ± 1094): wynik silnie zależy od losowania wag.
- Podział train/test jest **stały** (zawsze ten sam), a seed zmienia **tylko** wagi początkowe. Dzięki temu ± mierzy
  wyłącznie wpływ inicjalizacji.

---

## 11. Teoria części 2: dlaczego uczenie głębokich sieci jest trudne

### 11.1. Funkcja wypukła vs niewypukła

W lab2 (regresja logistyczna) funkcja kosztu była **wypukła**: wygląda jak jedna miska. Gdziekolwiek zaczniesz, idąc w dół
dojdziesz do tego samego dna (minimum globalnego). Nie ma pułapek.

Sieć z 2 lub więcej warstwami ma funkcję kosztu **niewypukłą**: góry, doliny, przełęcze, płaskowyże. Można by myśleć,
że największym problemem są **minima lokalne** (dołki, które nie są najgłębsze). Treść zadania (i badania z ostatnich 20 lat)
mówią, że w dużych sieciach to nie jest główny problem. Główne problemy to:

- **punkty siodłowe** – miejsca, gdzie gradient = 0, ale to nie jest minimum;
- **płaskowyże (plateau)** – duże obszary, gdzie gradient jest prawie zerowy.

### 11.2. Punkt siodłowy

Najprostszy przykład: f(x, y) = x² − y² (to dokładnie kształt z obrazka w PDF, siodło konia).
- W punkcie (0, 0) gradient = (2x, −2y) = (0, 0);
- w kierunku x to jest **minimum** (x² rośnie w obie strony);
- w kierunku y to jest **maksimum** (−y² maleje w obie strony).

Spadek gradientu, który trafi dokładnie w (0, 0), zatrzyma się, bo gradient jest zerowy, choć w kierunku y można iść dalej w dół.
Jeśli jest tylko **blisko** siodła, gradient jest bardzo mały i ucieczka trwa bardzo długo.

**Gdzie to jest w Twoim zadaniu?** Punkt „wszystkie wagi = 0” jest punktem siodłowym (rozdział 10.2). Inicjalizacja N(0, 0.01)
startuje tuż przy nim. Plateau z części 1 to właśnie powolna ucieczka z okolic siodła.

Dlaczego w wielu wymiarach siodła są częstsze niż minima? W punkcie z gradientem 0 funkcja może w każdym z tysięcy kierunków
rosnąć albo maleć. Żeby to było minimum, musi rosnąć we **wszystkich** kierunkach naraz. Przy tysiącach kierunków prawie zawsze
w którymś maleje, więc prawie każdy taki punkt jest siodłem.

### 11.3. Zanikające i eksplodujące gradienty

Z reguły łańcuchowej: gradient pierwszej warstwy = iloczyn czynników ze **wszystkich** warstw po drodze.
Jeśli każda warstwa mnoży gradient przez q:
- q = 0.037 (wagi 0.01): po 5 warstwach 0.037⁵ ≈ 7·10⁻⁸ → **gradient zanika** (*vanishing gradients*), wagi stoją;
- q = 3.7 (wagi 1): po 5 warstwach 3.7⁵ ≈ 700 → **gradient eksploduje** (*exploding gradients*), kroki są gigantyczne, uczenie się rozbiega;
- q = 1 (He): po dowolnej liczbie warstw gradient zachowuje skalę.

To jest mnożenie, więc efekt jest **wykładniczy** w liczbie warstw. Dlatego problem jest niewidoczny przy 1 warstwie,
a krytyczny przy 5. To uzasadnia wybór głębokości 5 w części 2: problemy muszą być widoczne (wymóg z treści).

### 11.4. Warunek stopu może oszukać

W lab2 kończyłeś uczenie, gdy koszt przestawał się zmieniać: |L_k − L_(k−1)| < tol. Dla funkcji wypukłej to sensowne:
koszt przestaje się zmieniać tylko blisko minimum. Ale na **płaskowyżu** koszt też prawie się nie zmienia, a do minimum jest daleko.
Taki warunek uzna „utknięcie” za „zbieżność”. W części 2 pokazujesz to liczbami (rozdział 14).

---

## 12. ADAM, tryby uczenia i inicjalizacja He: matematyka

### 12.1. Trzy tryby liczenia gradientu

Koszt to średnia po pacjentach, więc gradient też jest średnią gradientów pojedynczych pacjentów:

```
∇L = (1/N) · Σ_k ∇L_k
```

Możemy go liczyć z różnej liczby pacjentów:

| Tryb | Pacjentów na krok | Kroków na epokę (N = 242) | Gradient |
|---|---|---|---|
| pełny zbiór (*full batch*) | 242 | 1 | dokładny |
| paczki (*mini-batch*) | 32 | 8 (7 po 32 + 1 po 18) | przybliżony |
| pojedyncze przykłady (*SGD*) | 1 | 242 | bardzo zaszumiony |

**Przykład z życia:** chcesz znać średni wzrost w klasie. Możesz zmierzyć wszystkich (dokładnie, ale długo),
losowe 5 osób (prawie dobrze, szybko), albo jedną losową osobę (bardzo niedokładnie, natychmiast).
Gradient z paczki to „średnia z próbki”: **w średniej** jest równy prawdziwemu gradientowi, ale ma szum.

**Co daje mniejsza paczka:**
- **więcej kroków na epokę**: SGD robi 242 kroki tam, gdzie pełny zbiór robi 1. Na epokę uczy się więc dużo szybciej;
- **szum**: każdy krok idzie trochę w złą stronę. Szum może pomóc wyrwać się z siodła (gradient przypadkowo „pchnie” w kierunku ucieczki),
  ale też sprawia, że krzywa kosztu jest poszarpana;
- **koszt obliczeń**: 242 małe mnożenia są w Pythonie dużo wolniejsze niż jedno duże (narzut pętli, numpy jest wydajny
  na dużych macierzach). Dlatego SGD jest najwolniejszy w czasie rzeczywistym.

**Paczki** łączą zalety: dużo kroków, umiarkowany szum, wydajne mnożenie macierzy. To standard w praktyce.

**Tasowanie (shuffle):** w każdej epoce pacjenci są w innej, losowej kolejności. Bez tego paczki byłyby zawsze takie same
i sieć mogłaby się „przyzwyczaić” do kolejności.

### 12.2. Momentum: kulka, która się toczy

Zwykły GD robi krok w kierunku bieżącego gradientu. **Momentum** pamięta kierunek poprzednich kroków, jak ciężka kulka
tocząca się po zboczu: nabiera rozpędu i nie zatrzymuje się na małych płaskich fragmentach.

```
m = β1·m + (1 − β1)·g
```

To jest **średnia krocząca wykładniczo** gradientów: nowa średnia = 90% starej + 10% nowego gradientu (β1 = 0.9).
Efektywnie uśrednia około 1/(1 − β1) = 10 ostatnich gradientów.

Co to daje:
- **gradient stale w tę samą stronę** (np. g = 0.5, 0.5, 0.5…): m rośnie do 0.5, czyli kierunek się utrzymuje;
- **gradient skacze ±** (np. +1, −1, +1, −1, oscylacja w wąskim wąwozie): m się wyzerowuje, oscylacje się tłumią;
- **przy siodle**: nawet gdy gradient na chwilę maleje, m pamięta poprzedni kierunek, więc kulka „przejeżdża z rozpędu”.

### 12.3. Skalowanie przez wielkość gradientu (RMSProp)

Drugi pomysł: pamiętać, jak **duże** były ostatnio gradienty, i dzielić przez to krok:

```
v = β2·v + (1 − β2)·g²            (średnia z kwadratów gradientu, β2 = 0.999 → około 1000 ostatnich kroków)
krok = lr · g / √v
```

√v to „typowa wielkość” gradientu (pierwiastek ze średniego kwadratu, *RMS*). Dzielenie przez nią **normalizuje** krok:
- jeśli gradienty są malutkie (10⁻⁸), √v ≈ 10⁻⁸, więc g/√v ≈ 1 i krok ≈ lr. **Mały gradient dostaje normalny krok.**
  To jest zasada z treści: „jeśli ostatnie kroki były zbyt wolne, to teraz chcemy przyspieszyć”;
- jeśli gradienty są gigantyczne (10⁶), √v ≈ 10⁶, więc krok ≈ lr. **Wielki gradient nie wyrzuca wag w kosmos.**
- każda waga ma **własne** v, więc każda ma własny, dopasowany krok. To rozwiązuje też problem z danymi nieznormalizowanymi
  (każda cecha dostaje swoją skalę kroku).

### 12.4. ADAM = momentum + skalowanie

ADAM (*Adaptive Moment Estimation*) łączy oba pomysły. Dla **każdej wagi osobno**:

```
m = β1·m + (1 − β1)·g                   (1) średnia gradientu (kierunek, momentum)
v = β2·v + (1 − β2)·g²                  (2) średnia kwadratu gradientu (skala)
m̂ = m / (1 − β1ᵗ)                       (3) korekcja biasu
v̂ = v / (1 − β2ᵗ)                       (4) korekcja biasu
w = w − lr · m̂ / (√v̂ + ε)              (5) krok
```

Domyślne wartości: β1 = 0.9, β2 = 0.999, ε = 10⁻⁸, lr = 0.001. t to numer kroku (1, 2, 3, …).

**Literówki w PDF.** W treści zadania wzory wyglądają tak, jakby nawias był w złym miejscu `(1 − β1·∂L/∂w)` oraz jakby w v
nie było kwadratu. Poprawnie (oryginalna praca Kingma & Ba, 2014) jest `(1 − β1)·g` oraz `(1 − β2)·g²`. Bez kwadratu v byłoby
średnią gradientów, która może być **ujemna**, a pierwiastek z liczby ujemnej nie ma sensu. Do tego v nie mierzyłoby wielkości
gradientu, bo dodatnie i ujemne by się znosiły. Warto to powiedzieć na prezentacji, bo pokazuje, że rozumiesz, co robi v.

**Po co korekcja biasu?** m i v startują od 0. W pierwszym kroku m = 0.9·0 + 0.1·g = **0.1·g**, czyli 10 razy za mało
(średnia jest „ściągnięta” do zera przez wartość startową). Dzielenie przez (1 − β1¹) = 0.1 naprawia to: m̂ = g.
Z czasem βᵗ → 0 i korekcja przestaje działać (dzielenie przez około 1).

Przykład liczbowy (policzony w Pythonie), stały gradient g = 0.5:

| t | m | v | m̂ = m/(1−0.9ᵗ) | v̂ = v/(1−0.999ᵗ) | krok = 0.001·m̂/√v̂ |
|---|---|---|---|---|---|
| 1 | 0.05 | 0.00025 | 0.5 | 0.25 | 0.001 |
| 2 | 0.095 | 0.00050 | 0.5 | 0.25 | 0.001 |
| 3 | 0.1355 | 0.00075 | 0.5 | 0.25 | 0.001 |

Bez korekcji pierwszy krok byłby 0.001·0.05/√0.00025 = 0.0032 (3 razy za duży, bo v jest jeszcze mocniej zaniżone niż m).
Z korekcją m̂ i v̂ od razu są poprawnymi oszacowaniami g i g².

**Najważniejsza własność:** przy stałym gradiencie m̂/√v̂ = g/|g| = ±1, więc **krok ma długość lr, niezależnie od wielkości
gradientu**. Twój test to potwierdza: dla gradientu pomnożonego przez 10⁶ pierwszy krok i tak wynosi dokładnie 0.01 (= lr).

**Po co ε?** Żeby nie dzielić przez 0, gdy gradient jest zerowy. Ciekawy szczegół: ε = 10⁻⁸ jest tego samego rzędu co gradienty
przy inicjalizacji 0.01 (około 10⁻⁸). Dla g = 10⁻⁸: √v̂ = 10⁻⁸, więc krok = lr·10⁻⁸/(10⁻⁸ + 10⁻⁸) = **lr/2 = 0.0005**
(policzone w Pythonie). Czyli ε trochę hamuje ADAM przy ekstremalnie małych gradientach, ale nadal krok jest 10⁶ razy większy niż w GD.

**Wady ADAM** (żeby nie wyglądało, że jest magiczny):
- krok ograniczony do około lr. Jeśli wagi są „daleko” (np. inicjalizacja N(0, 1) i wagi trzeba zmienić o około 1),
  przy lr = 0.001 potrzeba setek–tysięcy kroków. Widać to w eksperymencie (rozdział 14.4);
- tak szybko schodzi z kosztem treningowym, że szybciej się przeucza;
- więcej pamięci (m i v dla każdej wagi).

### 12.5. Inicjalizacja: matematyka i He

**Wariancja jednego neuronu.** Neuron liczy z = Σ_{i=1..n} w_i·a_i. Załóżmy, że wagi są niezależne, mają średnią 0
i wariancję σ², a wejścia a_i mają średni kwadrat E[a²]. Wtedy:

```
Var(z) = Σ Var(w_i·a_i) = n · σ² · E[a²]
```

(Wariancja sumy niezależnych składników to suma wariancji, a Var(w·a) = E[w²]·E[a²] = σ²·E[a²], gdy w ma średnią 0.)

**Wpływ ReLU.** Jeśli z poprzedniej warstwy jest symetryczne wokół 0, ReLU zeruje połowę wartości, a dodatnie zostawia.
Średni kwadrat spada o połowę: E[a²] = ½·Var(z_poprzednie). Razem:

```
Var(z_l) = n · σ² · ½ · Var(z_{l−1})    →    odchylenie mnoży się przez  σ · √(n/2)  na warstwę
```

Dla n = 28, √(n/2) = √14 ≈ 3.74:

| σ (std wag) | Czynnik na warstwę σ·3.74 | Po 5 warstwach | Efekt |
|---|---|---|---|
| 0.01 | 0.037 | 0.037⁵ ≈ 7·10⁻⁸ | zanikanie |
| √(2/28) ≈ 0.267 (He) | **1.0** | 1 | skala zachowana |
| 1 | 3.74 | 3.74⁵ ≈ 730 | eksplozja |
| 10 | 37.4 | 37.4⁵ ≈ 7·10⁷ | gigantyczna eksplozja |

**Inicjalizacja He** (Kaiming He i in., 2015) wybiera σ tak, żeby czynnik był równy 1:

```
σ² · n / 2 = 1    →    σ = √(2 / n_in)
```

To jest dokładnie wzór z PDF. „2” bierze się z ReLU (które zabiera połowę wariancji). Dla aktywacji bez tego efektu
(np. tanh) analogiczny wzór to √(1/n) (inicjalizacja Xaviera/Glorota).

**Backward działa tak samo**: gradient przechodzi przez Wᵀ (dx = d·W), z czynnikiem σ·√(n_out/2). Przy warstwach tej samej
szerokości He zachowuje skalę i w forward, i w backward.

**Dlaczego n_in, a nie stała?** Bo pierwsza warstwa ma 28 wejść, a gdyby warstwa miała 512 wejść, suma 512 składników
byłaby większa, więc wagi muszą być mniejsze (√(2/512) ≈ 0.0625). He dopasowuje σ do każdej warstwy osobno.
To odróżnia He od „po prostu dobrej stałej”.

**Dlaczego bias = 0?** Bias nie wpływa na wariancję sumy (nie jest mnożony przez wejście). Zerowy bias to standard przy He
i dzięki temu porównanie inicjalizacji dotyczy tylko wag. Symetrię łamią wagi (losowe), więc zerowy bias nie szkodzi.

---

## 13. Kod `lab3_cz2.py` linijka po linijce

### 13.1. Importy i stałe

```python
# Ćwiczenie 3, część 2: ADAM, tryby uczenia (pełny zbiór / paczki / pojedyncze przykłady), inicjalizacja wag.
# Moduły sieci i dane bierzemy z lab3.py (sprawdzone testem gradientu) - import nie uruchamia eksperymentów części 1.
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from lab3 import (MLP, Linear, SigmoidBCELoss, X_train, X_test, y_train, y_test, n_features,
                  metrics, mean_std)
```

- `sys` – do odczytania argumentu z linii poleceń (`python lab3_cz2.py 0.05`, szybka próba).
- `time` – pomiar czasu przebiegów.
- `warnings` – wyciszenie ostrzeżeń numpy (niżej).
- **`from lab3 import ...`** – kluczowa decyzja architektoniczna. Część 2 **nie kopiuje** modułów sieci ani przygotowania danych,
  tylko je importuje. Zalety:
  - backward jest ten sam, który przeszedł test gradientu (rozdział 9), więc nie trzeba go sprawdzać drugi raz;
  - dane i podział train/test są identyczne jak w części 1;
  - jeśli coś poprawisz w module, poprawka działa w obu częściach.
  Ten import ustawia też 1 wątek numpy (bo `lab3.py` robi to na samej górze) i katalog roboczy.

```python
SEEDS = [0, 1, 2]
PLATEAU_LOSS = 0.6   # "koniec plateau" = pierwsza epoka z kosztem train < 0.6 (start: ln 2 ≈ 0.693), jak w części 1
TOL = 1e-6           # klasyczny warunek stopu z lab2: |L_k - L_(k-1)| < TOL
DIVERGE_FACTOR = 1e4 # rozbieżność = koszt train > 10^4 x koszt przed uczeniem (albo nan / inf)
HIDDEN = [n_features] * 5    # 5 warstw ukrytych po 28 neuronów (sieć "głęboka" dla tego zbioru)
LR_GD = 0.01                 # jak w części 1
LR_ADAM = 0.001              # standardowa wartość dla ADAM

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean po rozbieżnych przebiegach (same nan)
```

- `TOL` – próg warunku stopu z lab2. **Nie używamy go do zatrzymania uczenia**, tylko sprawdzamy, kiedy by zadziałał
  (żeby pokazać, że oszukuje na plateau).
- `DIVERGE_FACTOR = 1e4` – definicja rozbieżności. Uzasadnienie w 13.4.
- `HIDDEN = [28]*5` – 5 warstw ukrytych (rozdział 11.3: przy 5 warstwach efekty inicjalizacji są wyraźne).
- `LR_GD = 0.01` jak w części 1 (żeby wyniki były porównywalne), `LR_ADAM = 0.001` – domyślna wartość z oryginalnej pracy o ADAM.
  ADAM ma więc 10× mniejszy lr, a mimo to jest szybszy, więc jego przewaga nie bierze się z większego lr.
- `warnings.filterwarnings` – `np.nanmean` z kolumny złożonej z samych NaN (np. po rozbieżności wszystkich seedów) wypisuje
  ostrzeżenie „Mean of empty slice”. To jest oczekiwane, więc je wyciszamy, żeby nie zaśmiecało wyników.

### 13.2. Inicjalizacja wag

```python
def linear_layers(model):
    return [layer for layer in model.layers if isinstance(layer, Linear)]
```

Pomocnicza funkcja: lista samych warstw `Linear` (bez ReLU). To *list comprehension*: pętla, która od razu buduje listę.
Dla 5 warstw ukrytych zwraca 6 warstw Linear (5 ukrytych + wyjściowa).

```python
def init_weights(model, init, seed):
    # init: "he" -> N(0, sqrt(2 / d_in)), osobne odchylenie dla każdej warstwy
    #       liczba -> N(0, init) dla wszystkich warstw (np. 0.01, 1.0, 10.0)
    # bias = 0, żeby porównanie dotyczyło tylko rozkładu wag W
    rng = np.random.default_rng(seed)
    for layer in linear_layers(model):
        n_out, n_in = layer.W.shape
        std = np.sqrt(2 / n_in) if init == "he" else float(init)
        layer.W = rng.normal(0, std, (n_out, n_in))
        layer.b = np.zeros((1, n_out))
```

- `n_out, n_in = layer.W.shape` – rozpakowanie kształtu macierzy wag; n_in to liczba wejść warstwy (d_in z PDF).
- `std = ... if ... else ...` – wyrażenie warunkowe w jednej linii: dla `"he"` liczy √(2/n_in) (dla n_in = 28: 0.267),
  dla liczby bierze ją jako odchylenie.
- `rng.normal(0, std, ...)` – nowe wagi z N(0, std). Stare (z konstruktora MLP) są nadpisywane.
- `np.zeros(...)` – bias 0 (uzasadnienie w rozdziale 12.5).

**Dlaczego osobna funkcja zamiast zmiany `MLP` w `lab3.py`?** Bo `MLP` przyjmuje jedno `std` dla wszystkich warstw,
a He potrzebuje innego std w każdej warstwie. Zmiana `MLP` oznaczałaby zmianę zamkniętej i sprawdzonej części 1.
Losowanie wag „od nowa” po zbudowaniu sieci daje ten sam efekt bez ruszania części 1.

```python
def build_model(hidden, init, seed):
    model = MLP([n_features] + hidden + [1], seed=seed)
    init_weights(model, init, seed)
    return model
```

Buduje sieć (moduły z `lab3.py`) i nadpisuje wagi wybraną inicjalizacją. Ten sam seed = te same wagi.

### 13.3. Optymalizatory

```python
class GD:
    def __init__(self, lr=LR_GD):
        self.lr = lr

    def step(self, model):
        for layer in linear_layers(model):
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db
```

Zwykły spadek gradientu, ten sam wzór co `MLP.step` z części 1, ale jako osobna klasa. Dzięki temu pętla uczenia
nie wie, jakiego optymalizatora używa: woła tylko `optimizer.step(model)`. Zamiana GD na ADAM to zamiana jednego obiektu.

```python
class Adam:
    # m - średnia krocząca gradientu (momentum), v - średnia krocząca kwadratu gradientu.
    # Oba osobno dla każdej wagi, startują od zera, stąd korekcja biasu przez 1 - beta^t.
    def __init__(self, lr=LR_ADAM, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m, self.v = {}, {}
```

- `self.t` – licznik kroków (potrzebny do korekcji biasu β^t).
- `self.m, self.v` – **słowniki**: klucz = która macierz parametrów (np. `(0, "W")` = wagi warstwy 0), wartość = macierz m lub v
  tego samego kształtu co parametr. Czyli m i v są **osobno dla każdej wagi** (każdy element macierzy ma swoje m i v).

```python
    def update(self, key, param, grad):
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

Wzory (1)–(5) z rozdziału 12.4, linijka w linijkę:
- przy pierwszym użyciu danej macierzy tworzy m i v z zerami (`zeros_like` = zera o tym samym kształcie);
- `grad ** 2` – kwadrat **element po elemencie** (każda waga osobno), a nie mnożenie macierzy;
- `np.sqrt(v_hat)` – też element po elemencie;
- `param -= ...` – zmienia macierz **w miejscu**. To ważne: `param` to ta sama tablica co `layer.W`, więc zmiana trafia
  do sieci. (Gdyby było `param = param - ...`, powstałaby nowa tablica, a sieć by się nie zmieniła.)

```python
    def step(self, model):
        self.t += 1
        for i, layer in enumerate(linear_layers(model)):
            self.update((i, "W"), layer.W, layer.dW)
            self.update((i, "b"), layer.b, layer.db)
```

- `self.t += 1` raz na krok (przed aktualizacjami), więc w pierwszym kroku t = 1 i korekcja dzieli przez 1 − 0.9 = 0.1.
- Dla każdej warstwy Linear aktualizuje W i b, każdą ze swoimi m i v.
- **Nowy obiekt Adam dla każdego przebiegu** (w `experiment` jest `make_opt()`), więc m i v nie „przeciekają” między seedami.

### 13.4. Pętla uczenia

```python
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
```

Funkcja **pomiarowa** (nie uczy):
- robi forward ręcznie, warstwa po warstwie, żeby po każdej `Linear` zapisać **odchylenie standardowe wyjścia** (`x.std()`).
  To jest wykres „stan sieci przed uczeniem” (pokazuje zanikanie/eksplozję sygnału);
- liczy koszt na **całym** zbiorze treningowym;
- robi backward na całym zbiorze (bez kroku uczenia), żeby zmierzyć **pełny gradient** |dW| każdej warstwy.
  `np.linalg.norm` to norma Frobeniusa: √(suma kwadratów wszystkich elementów macierzy), czyli „długość” gradientu.

Dlaczego pełny gradient, a nie ten z ostatniej paczki? Przy SGD gradient z jednego pacjenta jest bardzo zaszumiony.
Pełny gradient jest porównywalny między trybami.

```python
def train(hidden, init, optimizer, batch_size=None, epochs=1000, seed=0):
    model = build_model(hidden, init, seed)
    loss_fn = SigmoidBCELoss()
    rng = np.random.default_rng(seed + 1000)          # osobny generator do tasowania przykładów
    n = len(X_train)
    bs = n if batch_size is None else batch_size
    h = {"train_loss": [], "test_loss": [], "f1": [], "steps": [], "grad_norms": [], "diverged_at": None}
    steps = 0
```

- `optimizer` – obiekt GD albo Adam.
- `rng` z `seed + 1000` – **osobny** generator do tasowania pacjentów. Gdyby był ten sam co do wag, tasowanie zależałoby od
  liczby wylosowanych wag. Tak: wagi zależą od `seed`, kolejność od `seed + 1000`, wszystko powtarzalne.
- `bs` – rozmiar paczki: `None` oznacza pełny zbiór (bs = 242).
- `h` – historia: koszty, F1, liczba kroków, normy gradientów i epoka rozbieżności.

```python
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
```

- Funkcja **zagnieżdżona** (zdefiniowana wewnątrz `train`), więc ma dostęp do `model`, `h`, `steps`.
- Zapisuje stan sieci: koszt train (pełny zbiór), koszt test, F1 (ten sam ręczny wzór co w części 1), liczbę kroków, normy gradientu.
- **Indeks 0 = przed uczeniem.** Dzięki temu wszystkie krzywe (różne tryby, różne optymalizatory) startują z tego samego punktu.
  W części 1 tego nie było: `train_loss[0]` był kosztem w pierwszej epoce.

```python
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
```

- `np.errstate(all="ignore")` – przy inicjalizacji N(0, 10) liczby przekraczają zakres float64 (inf, NaN) i numpy wypisywałby
  setki ostrzeżeń. Wyciszamy je, bo rozbieżność wykrywamy sami (niżej).
- `h["init_stds"] = evaluate()` – pomiar przed uczeniem (epoka 0), zapisuje też odchylenia wyjść warstw do wykresu.
- `rng.permutation(n)` – losowa kolejność 0…241. Przy pełnym zbiorze tasowanie nie ma sensu (i tak bierzemy wszystkich), więc `np.arange(n)`.
- `for start in range(0, n, bs)` – początki paczek: dla bs = 32: 0, 32, 64, …, 224. Ostatnia paczka (224:256) ma tylko 18 pacjentów,
  bo `order[224:256]` ucina się do końca tablicy. Dla bs = 1: 242 kroki. Dla bs = 242: jeden.
- `X_train[idx]` – wybiera wiersze (pacjentów) z paczki.
- **forward → backward → step** na paczce. Koszt jest średnią po pacjentach **w paczce** (bo `SigmoidBCELoss` dzieli przez `len(y)`),
  więc gradient ma tę samą skalę niezależnie od rozmiaru paczki.
- `evaluate()` po każdej epoce: pomiar na pełnych zbiorach.

```python
            # rozbieżność: koszt nieskończony albo DIVERGE_FACTOR razy większy niż przed uczeniem.
            # Sam test na nan nie wystarcza - stabilny wzór kosztu daje skończone, ale absurdalne wartości (np. 1e287)
            loss = h["train_loss"][-1]
            if not np.isfinite(loss) or loss > DIVERGE_FACTOR * max(h["train_loss"][0], 1.0):
                h["diverged_at"] = epoch
                break
```

**Wykrywanie rozbieżności** – ważna decyzja, warto ją umieć uzasadnić:
- `np.isfinite` – False dla inf i NaN;
- dlaczego nie wystarczy sprawdzić NaN? Stabilny wzór kosztu (rozdział 6.7) **nie produkuje NaN** nawet dla gigantycznych logitów:
  dla z = 10²⁸⁷ koszt wynosi po prostu około 10²⁸⁷, liczba skończona. W pierwszej wersji skryptu przebiegi z kosztem 10²⁸⁷
  nie były uznawane za rozbieżne i psuły średnie i wykresy;
- dlaczego „10⁴ razy więcej niż przed uczeniem”, a nie stały próg? Bo przy N(0, 10) koszt **startowy** to już 7·10⁸. Stały próg
  (np. 1000) uznałby za rozbieżny nawet przebieg ADAM, który powoli schodzi z 7·10⁸ w dół. Względny próg mówi:
  „koszt urósł 10 000 razy od startu, więc uczenie idzie w złą stronę”;
- `max(..., 1.0)` – dla dobrej inicjalizacji start to około 0.7; wtedy próg to 10⁴·1 = 10 000;
- `break` – przerywa uczenie (dalsze epoki nic by nie dały, a liczby i tak są nieskończone).

```python
    # po rozbieżności dopełniamy historię wartością nan, żeby wszystkie przebiegi miały tę samą długość
    for k in ["train_loss", "test_loss", "f1", "steps"]:
        h[k] = np.array(h[k] + [np.nan] * (epochs + 1 - len(h[k])), dtype=float)
    norms = np.array(h["grad_norms"])
    h["grad_norms"] = np.vstack([norms, np.full((epochs + 1 - len(norms), norms.shape[1]), np.nan)])
    h["steps"] = np.arange(epochs + 1) * int(np.ceil(n / bs))
    return model, h
```

- Przebieg przerwany w epoce 4 ma 5 wartości zamiast 2001. Żeby uśredniać 3 seedy (np. `np.stack`), wszystkie historie muszą mieć
  tę samą długość, więc dopełniamy NaN. NaN na wykresie to „brak punktu”, a `np.nanmean` je pomija.
- `h["steps"]` – liczba kroków po każdej epoce: epoka · (kroków na epokę). `np.ceil(242/32) = 8`. Do osi X „liczba kroków”.

### 13.5. Podsumowanie przebiegu

```python
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
```

- `metrics` z części 1 (accuracy, precision, recall, F1 na teście dla końcowego modelu).
- `plateau_end` – pierwsza epoka z kosztem < 0.6. Bez `+ 1`, bo tu indeks 0 to „przed uczeniem”, więc indeks = numer epoki.
- `tl[~np.isnan(tl)][-1]` – ostatnia **nie-NaN** wartość (dla przerwanych przebiegów ostatnia przed przerwaniem).
- `np.nanmin` – minimum z pominięciem NaN.

```python
    # gdzie zatrzymałby się warunek stopu z lab2 (|zmiana kosztu| < TOL) i jaki byłby wtedy koszt
    small = np.nonzero(np.abs(np.diff(tl)) < TOL)[0]
    m["tol_stop_epoch"] = small[0] + 1 if len(small) else np.nan
    m["tol_stop_loss"] = tl[small[0] + 1] if len(small) else np.nan
```

- `np.diff(tl)` – różnice między kolejnymi epokami: [L1 − L0, L2 − L1, …]. Element k to zmiana w epoce k + 1.
- Pierwsza epoka, w której |zmiana| < 10⁻⁶, to miejsce, gdzie warunek z lab2 **zatrzymałby** uczenie. Zapisujemy też koszt w tym momencie.
  Jeśli koszt wtedy wynosi 0.69, to warunek uznał utknięcie za zbieżność.

```python
    if h["diverged_at"]:
        # koszt rozbieżnego przebiegu (np. 1e11) zniekształciłby średnią pozostałych seedów -
        # pomijamy go w średniej, liczba rozbieżnych przebiegów jest wypisywana osobno
        for k in ["train_loss", "test_loss", "min_test_loss"]:
            m[k] = np.nan
    return m
```

Przykład, dlaczego to potrzebne: przy N(0, 1) dwa seedy kończą z kosztem około 0.02 i 0.55, a rozbieżny z 10¹¹.
Średnia z trzech: 3·10¹⁰, co nic nie mówi. Średnia z dwóch nierozbieżnych: 0.284, plus informacja „1/3 rozbieżne”.
F1 rozbieżnego seeda **zostaje** w średniej (F1 = 0 to prawdziwy wynik tego modelu: nie działa).

### 13.6. Eksperyment i wykresy

```python
def fmt(agg, k, p=".3g"):
    mean, std = agg[k]
    return "—" if np.isnan(mean) else f"{mean:{p}} ± {std:{p}}"
```

Formatowanie „średnia ± odchylenie”. `.3g` = 3 cyfry znaczące w formacie „ogólnym”: 0.0445 albo 7.4e+08, bo koszty
w części 2 mają skrajnie różne rzędy wielkości. NaN → „—”.

```python
def experiment(configs, title, filename, epochs, by_steps=False):
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
```

- `make_opt` – **klasa** (GD albo Adam), a nie obiekt. `make_opt()` tworzy **nowy** optymalizator dla każdego seeda.
  To ważne dla ADAM: każdy przebieg musi zaczynać z m = v = 0 i t = 0.
- `runs` – 3 przebiegi (model, historia); `ms` – ich podsumowania; `agg` – średnia ± odchylenie każdej metryki.
- `n_stuck` – ile seedów nie wyszło z plateau; `n_div` – ile się rozbiegło.

```python
        print(f"{name:24s} koniec plateau: {fmt(agg, 'plateau_end', '.0f')} (nie wyszło: {n_stuck}/{len(SEEDS)})  "
              f"rozbieżność: {n_div}/{len(SEEDS)} (epoka {fmt(agg, 'diverged_at', '.0f')})  "
              f"stop |ΔL|<{TOL:g}: epoka {fmt(agg, 'tol_stop_epoch', '.0f')}, koszt {fmt(agg, 'tol_stop_loss')}  "
              f"[{(time.time() - t) / len(SEEDS):.1f} s / przebieg]")
        print(f"{'':24s} koszt train = {fmt(agg, 'train_loss')}  koszt test = {fmt(agg, 'test_loss')}  "
              f"min koszt test = {fmt(agg, 'min_test_loss')}  accuracy = {fmt(agg, 'accuracy', '.3f')}  "
              f"F1 = {fmt(agg, 'f1', '.3f')}  [F1 dla seedów: {', '.join(f'{m['f1']:.3f}' for m in ms)}]")

    plot_experiment(results, title, filename, epochs, by_steps)
    return results
```

Wypisuje dwie linijki na konfigurację (to jest `wyniki_cz2.txt`) i rysuje wykres.

```python
def band(ax, x, curves, label, log=False):
    curves = np.where(np.isfinite(curves) & (np.abs(curves) < 1e30), curves, np.nan)   # bez inf na wykresie
    mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
    line, = ax.plot(x, mean, label=label)
    low = np.maximum(mean - std, mean * 0.1) if log else mean - std    # na skali log pas nie może zejść do <= 0
    ax.fill_between(x, low, mean + std, color=line.get_color(), alpha=0.2)
```

Rysuje linię średniej i pas odchylenia (jak w części 1), z dwoma zabezpieczeniami:
- `np.where(...)` – wartości nieskończone i większe niż 10³⁰ zamienia na NaN. Matplotlib wywala się, próbując narysować inf na osi log
  (to był błąd przy pierwszej próbie);
- na osi logarytmicznej nie da się narysować liczby ≤ 0, a średnia − odchylenie może wyjść ujemna (np. 0.3 − 0.5).
  `np.maximum(mean − std, mean·0.1)` ogranicza dół pasa do 10% średniej.

```python
def plot_experiment(results, title, filename, epochs, by_steps):
    panels = [("train_loss", "koszt (train)"), ("test_loss", "koszt (test)"), ("f1", "F1 (test)"),
              ("grad1", "|dW| warstwy 1 (pełny gradient)")]
    if by_steps:
        panels.insert(1, ("train_loss_steps", "koszt (train)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
    x = np.arange(epochs + 1)
    for name, (agg, hs) in results.items():
        for ax, (key, _) in zip(axes, panels):
            if key == "grad1":
                band(ax, x, np.stack([h["grad_norms"][:, 0] for h in hs]), name, log=True)
            elif key == "train_loss_steps":
                band(ax, hs[0]["steps"], np.stack([h["train_loss"] for h in hs]), name, log=True)
            else:
                band(ax, x, np.stack([h[key] for h in hs]), name, log=key != "f1")
```

- 4 panele: koszt train, koszt test, F1, **norma gradientu pierwszej warstwy** (`grad_norms[:, 0]` = kolumna 0 = warstwa 1,
  bo to ona najbardziej cierpi na zanikanie/eksplozję).
- Dla trybów uczenia (`by_steps=True`) dochodzi panel „koszt względem liczby kroków” (wstawiony jako drugi). Oś X to wtedy
  `steps`, a nie epoki: pozwala porównać tryby uczciwie „na krok” (rozdział 14.5).

```python
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_xscale("symlog", linthresh=1)          # symlog: oś log, ale epoka 0 (przed uczeniem) jest widoczna
        ax.set_xlabel("liczba kroków uczenia (symlog)" if key == "train_loss_steps" else "epoka (symlog)")
        ax.set_ylabel(ylabel)
        if key != "f1":
            ax.set_yscale("log")
        ax.grid(which="both", alpha=0.4)
        ax.legend(fontsize=8)
    plt.suptitle(f"{title}\nlinia = średnia z {len(SEEDS)} seedów, pas = ± odchylenie standardowe; "
                 f"oś Y kosztu w skali log")
    plt.tight_layout()
    plt.savefig(f"wykresy/{filename}")
    plt.close()
```

- **`symlog`** (symetryczny logarytm): liniowy w okolicy 0 (od 0 do `linthresh = 1`), logarytmiczny dalej. Dzięki temu
  **epoka 0** (przed uczeniem) jest widoczna, a reszta ma zalety skali log. W części 1 rozwiązaliśmy to inaczej (oś od 1).
- **Oś Y logarytmiczna** dla kosztów i gradientów: koszty w części 2 mają zakres od 10⁻⁶ do 10²⁹. Na zwykłej osi wszystko oprócz
  największej wartości byłoby płaską linią na dole.

```python
def plot_init_layers(results, filename):
    # odchylenie wyjścia i norma gradientu każdej warstwy PRZED uczeniem (średnia z seedów)
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
```

Wykres „stanu sieci przed uczeniem”: oś X to numer warstwy, oś Y (log) to odchylenie wyjścia i norma gradientu.
`grad_norms[0]` = wiersz 0 = pomiar przed uczeniem. `"o-"` = punkty połączone linią. Na skali log stały czynnik na warstwę
(np. ×3.7) wygląda jak **prosta linia**, więc od razu widać, czy sygnał rośnie, maleje, czy jest stały.

### 13.7. Test ADAM

```python
def test_adam():
    # f(w) = (w - 3)^2, start w = 5. Dwie własności ADAM:
    #  - pierwszy krok ma długość lr niezależnie od wielkości gradientu (m_hat / sqrt(v_hat) = g / |g| = ±1)
    #  - po wielu krokach w -> 3
    print("=== Test ADAM na f(w) = (w - 3)^2, start w = 5, lr = 0.01 ===")
    for scale in [1.0, 1e6]:                          # ten sam problem, gradient przemnożony przez 10^6
        opt = Adam(lr=0.01)
        w = np.array([5.0])
        for t in range(1, 3001):
            opt.t = t
            opt.update("w", w, scale * 2 * (w - 3))
            if t == 1:
                first_step = 5.0 - w[0]
        print(f"skala gradientu {scale:6.0e}: pierwszy krok = {first_step:.5f}, w po 3000 krokach = {w[0]:.4f}")
```

Test jednostkowy ADAM, analogiczny do testu gradientu z części 1, ale prostszy:
- funkcja (w − 3)², gradient 2(w − 3), minimum w = 3;
- `w = np.array([5.0])` – tablica, a nie zwykła liczba, bo `update` zmienia parametr **w miejscu** (`-=`), a to działa tylko na tablicach;
- `opt.t = t` – ustawiamy licznik ręcznie, bo nie wołamy `step(model)`, tylko bezpośrednio `update`;
- `scale = 10⁶` – ten sam problem z gradientem pomnożonym przez milion (funkcja 10⁶·(w − 3)², to samo minimum).

Wynik: w obu przypadkach **pierwszy krok = 0.01000** (dokładnie lr) i w = 3.0000. Zwykły GD przy gradiencie 10⁶ razy większym
zrobiłby krok 10⁶ razy większy i natychmiast się rozbiegł. To jest własność, która wyjaśnia wszystkie wyniki ADAM w eksperymentach.

### 13.8. Funkcja `main`

```python
def main(scale=1.0):
    # scale < 1 - szybka próba ze zmniejszoną liczbą epok (sprawdzenie, czy wszystko działa, i pomiar czasu)
    ep = lambda e: max(5, int(e * scale))
    t0 = time.time()
    test_adam()
```

- `scale` – mnożnik liczby epok. `python lab3_cz2.py 0.05` puszcza całość na 5% epok (kilkanaście sekund). Służy do sprawdzenia,
  czy kod działa i ile potrwa pełne uruchomienie, zanim puścimy coś na kilka minut.
- `lambda e: ...` – krótka funkcja bez nazwy: liczba epok przeskalowana, ale co najmniej 5.

```python
    # EKSPERYMENT 1 - INICJALIZACJA (zwykły GD, pełny zbiór)
    res_init = experiment(
        {
            "init N(0, 0.01)": (HIDDEN, 0.01, GD, None),
            "init He": (HIDDEN, "he", GD, None),
            "init N(0, 1)": (HIDDEN, 1.0, GD, None),
            "init N(0, 10) (zepsuta)": (HIDDEN, 10.0, GD, None),
        },
        "Inicjalizacja wag (5 warstw ukrytych, GD lr = 0.01, pełny zbiór)", "cz2_inicjalizacja.png", ep(2000),
    )
    plot_init_layers(res_init, "cz2_inicjalizacja_warstwy.png")
```

Porównanie inicjalizacji przy **wszystkim innym stałym** (zwykły GD, lr 0.01, pełny zbiór, 2000 epok). Jedyna różnica to rozkład wag,
więc różnice w wynikach wynikają tylko z inicjalizacji. Treść wymaga: std 1, He, zepsuta. Dodane 0.01 łączy wyniki z częścią 1
i pokazuje drugi rodzaj „zepsucia” (zanikanie).

```python
    # EKSPERYMENT 2 - GD vs ADAM (pełny zbiór)
    experiment(
        {
            "GD, init 0.01": (HIDDEN, 0.01, GD, None),
            "ADAM, init 0.01": (HIDDEN, 0.01, Adam, None),
            "GD, init He": (HIDDEN, "he", GD, None),
            "ADAM, init He": (HIDDEN, "he", Adam, None),
            "GD, init 1": (HIDDEN, 1.0, GD, None),
            "ADAM, init 1": (HIDDEN, 1.0, Adam, None),
        },
        f"GD (lr = {LR_GD}) vs ADAM (lr = {LR_ADAM}), 5 warstw ukrytych, pełny zbiór", "cz2_gd_vs_adam.png", ep(2000),
    )
```

GD vs ADAM w trzech sytuacjach: zanikające gradienty (0.01), dobra inicjalizacja (He), eksplodujące gradienty (1).
Pary różnią się tylko optymalizatorem. (N(0, 10) pominięte, bo GD rozbiega się w 1. epoce, a to już widać w eksperymencie 1.)

```python
    # EKSPERYMENT 3 - TRYBY UCZENIA (zwykły GD), osobno dla inicjalizacji He i 0.01 (plateau)
    for init, label, fname in [("he", "He", "cz2_tryby_he.png"), (0.01, "N(0, 0.01)", "cz2_tryby_001.png")]:
        experiment(
            {
                "pełny zbiór (242)": (HIDDEN, init, GD, None),
                "paczki po 32": (HIDDEN, init, GD, 32),
                "pojedyncze przykłady (1)": (HIDDEN, init, GD, 1),
            },
            f"Tryby uczenia, init {label} (5 warstw ukrytych, GD lr = {LR_GD})", fname, ep(300), by_steps=True,
        )
    print(f"\nCałkowity czas: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
```

- Tryby uczenia ze **zwykłym GD** (żeby widzieć efekt samego trybu, a nie ADAM).
- Dwie wersje: z He (normalne uczenie) i z 0.01 (sprawdzamy, czy szum z paczek wyciąga z płaskowyżu, co sugeruje treść).
- **300 epok**, a nie 2000, bo SGD jest wolny (około 6 s na 300 epok na jeden przebieg, około 40 s na 2000). Przy 300 epokach
  SGD i tak robi 72 600 kroków.
- `sys.argv` – lista argumentów z linii poleceń; `sys.argv[1]` to pierwszy argument (np. "0.05"). Bez argumentu `scale = 1`.

---

## 14. Wyniki części 2: co wyszło i dlaczego

### 14.1. Test ADAM

| Skala gradientu | Pierwszy krok | w po 3000 krokach |
|---|---|---|
| 1 | 0.01000 | 3.0000 |
| 10⁶ | 0.01000 | 3.0000 |

Implementacja jest poprawna (dochodzi do minimum) i ma kluczową własność: długość kroku nie zależy od skali gradientu.

### 14.2. Stan sieci przed uczeniem (wykres `cz2_inicjalizacja_warstwy.png`)

Zmierzone odchylenie wyjścia kolejnych warstw (seed 0):

| init | Warstwa 1 | 2 | 3 | 4 | 5 | 6 (wyjście) |
|---|---|---|---|---|---|---|
| 0.01 | 0.037 | 0.0013 | 5·10⁻⁵ | 1.8·10⁻⁶ | 4·10⁻⁸ | 8.5·10⁻¹⁰ |
| He | 0.98 | 0.95 | 0.96 | 0.90 | 0.55 | 0.31 |
| 1 | 3.7 | 13 | 50 | 180 | 410 | 850 |
| 10 | 37 | 1300 | 5·10⁴ | 1.8·10⁶ | 4·10⁷ | 8.5·10⁸ |

Porównaj z teorią z rozdziału 12.5: czynnik na warstwę σ·√14 = 0.037 / 1.0 / 3.7 / 37. **Pomiar zgadza się z teorią.**
Na wykresie (oś log) są to proste linie: w dół (0.01), poziomo (He), w górę (1 i 10).

Dlaczego dla He ostatnie warstwy mają trochę mniej (0.55, 0.31)? Teoria zakłada idealną symetrię i niezależność.
Przy 28 neuronach to tylko przybliżenie, a ostatnia warstwa ma 1 neuron (jedna liczba zamiast średniej z wielu). Rząd wielkości (około 1) jest zachowany.

**Ciekawy szczegół: przy 0.01 gradient |dW| jest podobny w każdej warstwie (około 10⁻⁸).** Można by oczekiwać, że gradient
zanika „w stronę wejścia”. Ale dW = dᵀ·x ma dwa czynniki:
- w warstwach blisko wyjścia: d (gradient z tyłu) jest duży, ale x (wejście warstwy) jest malutkie, bo sygnał już osłabł w forward;
- w warstwach blisko wejścia: x jest normalne (dane), ale d jest malutkie, bo osłabło w backward.

Iloczyn wychodzi podobny, ale w każdej warstwie bardzo mały. **Cała sieć jest zablokowana równomiernie.**

### 14.3. Eksperyment 4: inicjalizacja (GD, 2000 epok)

| init | Koniec plateau | Rozbieżność | Stop \|ΔL\| < 10⁻⁶ | Koszt train | F1 |
|---|---|---|---|---|---|
| N(0, 0.01) | nie wyszła | 0/3 | epoka 569, koszt 0.690 | 0.690 | 0.000 |
| He | 107 ± 48 | 0/3 | — | 0.045 | 0.784 ± 0.037 |
| N(0, 1) | 126 ± 98 (2 seedy) | 1/3 | epoka 265, koszt 0.37 | 0.284 (2 seedy) | 0.444 ± 0.316 |
| N(0, 10) | nie wyszła | 3/3 w 1. epoce | — | — | 0.419 ± 0.297 |

**N(0, 0.01) – zanikanie.** Gradient 10⁻⁸ · lr 0.01 = krok 10⁻¹⁰. Żeby waga zmieniła się o 0.01, potrzeba 10⁸ kroków.
Koszt spada minimalnie (0.6931 → 0.690, to tylko bias wyjścia, rozdział 14.6) i stoi.
**Warunek stopu z lab2 zatrzymałby uczenie w epoce 569 i ogłosił zbieżność.** To najlepsza ilustracja, dlaczego na płaskowyżu
„koszt przestał się zmieniać” nie znaczy „doszliśmy do minimum”.

**He – działa.** Wychodzi z plateau średnio w 107. epoce, koszt treningowy spada do 0.045. Jedyna inicjalizacja, która zadziałała
dla wszystkich seedów. Jest pewne plateau (107 epok), bo sieć ma 6 warstw i startowe gradienty (około 0.2) przy lr = 0.01
nie są duże, ale to nic w porównaniu z 0.01.

**N(0, 1) – eksplozja, wynik zależy od szczęścia.**
- koszt startowy około 750 (logity rzędu setek, sigmoid nasycony, a pewne błędne odpowiedzi kosztują setki, rozdział 3.2);
- gradienty około 10²–10³ na warstwę, czyli około 10 na pojedynczą wagę. Krok GD = 0.01·10 = 0.1 na wagę, przy wagach rzędu 1.
  Kroki są za duże, więc w pierwszych epokach koszt **rośnie** zamiast maleć (przeskakiwanie doliny);
- seed 1 rozbiegł się w 4. epoce (koszt przekroczył 10⁴ razy wartość startową). Pozostałe dwa „wygrzebały się”:
  przypadkowo trafiły w obszar, gdzie wagi się zmniejszyły i uczenie się ustabilizowało;
- koszt testowy około 2.3 przy min 0.41 dla He: sieć, która przetrwała, ma duże wagi i jest „zbyt pewna siebie”.

**N(0, 10) – natychmiastowa rozbieżność.** Gradient około 10⁶ na wagę, krok około 10⁴. Wagi wylatują w kosmos, koszt 10²⁹.
F1 = 0.629 dla dwóch seedów to **artefakt**: zepsuta sieć mówi „chory” dla każdego pacjenta, a wtedy F1 = 56/89 = 0.629
(rozdział 8.9). Gdyby prowadzący zapytał „to czemu zepsuta inicjalizacja ma F1 0.42?”: bo F1 nagradza mówienie „chory”
każdemu, kiedy chorych jest dużo. Dlatego patrzymy na koszt i rozbieżność, a nie na samo F1.

### 14.4. Eksperyment 5: GD vs ADAM (2000 epok)

| | Koniec plateau | Rozbieżność | Koszt train | Koszt test | Min. koszt test | F1 |
|---|---|---|---|---|---|---|
| GD, 0.01 | nie wyszła | 0/3 | 0.690 | 0.690 | 0.690 | 0.000 |
| ADAM, 0.01 | **67 ± 7** | 0/3 | 0.040 | 3.94 | 0.463 | 0.764 |
| GD, He | 107 ± 48 | 0/3 | 0.045 | 0.81 | 0.409 | 0.784 |
| ADAM, He | **15 ± 6** | 0/3 | 0.000005 | 2.97 | 0.407 | 0.756 |
| GD, 1 | 126 ± 98 | 1/3 | 0.284 | 2.56 | 2.31 | 0.444 |
| ADAM, 1 | 324 ± 103 | **0/3** | 0.00002 | 75 | 74.3 | 0.659 |

**ADAM wychodzi z płaskowyżu (0.01): 67 epok vs nigdy.** Mechanizm (rozdział 12.3–12.4): gradient 10⁻⁸ jest dzielony przez
√v̂ ≈ 10⁻⁸, więc krok ma długość około lr/2 do lr, czyli 0.0005–0.001, zamiast 10⁻¹⁰ jak w GD. To **10⁶–10⁷ razy większy** krok.
Na wykresie normy gradientu (ostatni panel) widać, co się dzieje: ADAM przez pierwsze kilkadziesiąt epok **powiększa wagi**
(każdy krok około 0.001 w spójnym kierunku, momentum utrzymuje kierunek), więc sygnał przestaje zanikać i |dW| rośnie z 10⁻⁸
do około 10⁻² w ciągu około 100 epok. Dopiero wtedy koszt zaczyna spadać. ADAM nie „naprawia” zanikających gradientów
matematycznie, tylko je **ignoruje**: krok zależy od kierunku gradientu, a nie od jego wielkości.

**ADAM przy He: 15 epok zamiast 107.** Przy dobrej inicjalizacji ADAM jest po prostu szybszy: każda waga ma swój dopasowany krok,
a momentum przyspiesza ruch w spójnym kierunku. Schodzi z kosztem treningowym do 5·10⁻⁶, czyli praktycznie zapamiętuje
wszystkich 242 pacjentów. Stąd koszt testowy 2.97 (przeuczenie).

**ADAM przy N(0, 1): nie rozbiega się, ale…**
- GD rozbiegł się w 1/3 przebiegów, ADAM w 0/3. Krok ADAM ≈ lr = 0.001 niezależnie od gradientu 10³, więc wagi nie mogą „wylecieć”;
- ale ta sama własność sprawia, że ADAM jest **wolny**: wagi rzędu 1 trzeba zmniejszyć, a przy kroku 0.001 to setki kroków.
  Koniec plateau dopiero w 324. epoce, później niż GD z tą samą inicjalizacją;
- **koszt testowy 75 – dziwny przypadek, warto umieć go wytłumaczyć.** Sieć kończy z dużymi wagami, więc logity są ogromne
  (setki). Na treningu wszystkie są po dobrej stronie, więc koszt ≈ 0. Na teście sieć myli się dla około 1/3 pacjentów (F1 0.66),
  a każda pomyłka jest „pewna na 100%”: logit np. −200 dla chorego pacjenta daje koszt około 200 (rozdział 3.2: koszt nieograniczony).
  Średnia z 61 pacjentów wychodzi około 75. **Wysoki koszt testowy przy niezłym F1 = model pewny siebie i często w błędzie.**
- Wniosek: **ADAM chroni przed rozbieżnością, ale nie naprawia złej inicjalizacji.**

**Minimum kosztu testowego jest podobne dla ADAM i GD przy He (0.407 vs 0.409).** ADAM nie znajduje „lepszego” modelu,
tylko dochodzi do niego szybciej i szybciej przechodzi w przeuczenie.

### 14.5. Eksperyment 6: tryby uczenia z inicjalizacją He (300 epok)

| Tryb | Kroków / epokę | Koniec plateau (epoka) | Koszt train | Koszt test | Min. koszt test | Czas przebiegu |
|---|---|---|---|---|---|---|
| pełny zbiór | 1 | 107 ± 48 | 0.393 | 0.454 | 0.454 | 0.3 s |
| paczki 32 | 8 | 14 ± 6 | 0.031 | 0.96 | 0.400 | 0.5 s |
| SGD | 242 | 1 ± 0 | 0.00003 | 3.01 | 0.413 | 6.3 s |

**Na epokę: SGD ≫ paczki ≫ pełny zbiór.** W jednej epoce SGD robi 242 kroki, paczki 8, pełny zbiór 1. Więcej kroków = szybszy postęp.
SGD wychodzi z plateau już w pierwszej epoce (czyli w pierwszych 242 krokach).

**Na krok: różnice są dużo mniejsze.** Na panelu „koszt względem liczby kroków” krzywe leżą blisko siebie, a na osi epok
były rozsunięte o rzędy wielkości. Pełny zbiór kończy się już przy 300 krokach (300 epok), a paczki i SGD po prostu „jadą dalej”,
bo w tych samych 300 epokach zrobiły 2400 i 72 600 kroków. Wniosek: przewaga małych paczek „na epokę” bierze się głównie
z **liczby kroków**, a nie z „lepszego” kierunku. Gradient z 32 pacjentów prowadzi w podobną stronę co z 242.

**Szum SGD.** Krzywa SGD ma skoki (około 20.–40. epoki), bo pojedynczy pacjent może pchnąć wagi w złą stronę.
Pełny gradient spada gładko.

**Odchylenie między seedami.** Pełny zbiór: koniec plateau 107 ± 48, szeroki pas, bo moment wyjścia z plateau zależy od
wylosowanych wag. SGD: 1 ± 0, bo setki zaszumionych kroków szybko „rozmywają” różnice startowe.

**Czas.** SGD jest około 20× wolniejszy na epokę (6.3 s vs 0.3 s), bo 242 małe operacje w Pythonie kosztują więcej niż
jedna duża (narzut pętli i wywołań, numpy jest zoptymalizowany pod duże macierze). Paczki kosztują prawie tyle co pełny zbiór,
a uczą się dużo szybciej, więc **paczki są najlepszym kompromisem** i standardem w praktyce.

**Przeuczenie.** SGD po 300 epokach (72 600 kroków) zapamiętał zbiór treningowy (koszt 0.00003), koszt testowy 3.0.
Minimum kosztu testowego jest podobne dla wszystkich trybów (0.40–0.45). Tryby różnią się **tempem**, nie „jakością” rozwiązania.

### 14.6. Eksperyment 6: tryby uczenia z inicjalizacją 0.01 – czy szum wyciąga z płaskowyżu?

| Tryb | Koniec plateau | Stop \|ΔL\| < 10⁻⁶ | Koszt train | F1 |
|---|---|---|---|---|
| pełny zbiór | nie wyszła | — | 0.690 | 0 |
| paczki 32 | nie wyszła | epoka 82 | 0.690 | 0 |
| SGD | nie wyszła | epoka 9 | 0.690 | 0 |

**Nie, szum nie wystarczył.** Treść mówi, że paczki „odrobinę mogą pomóc”, bo szum może wytrącić z siodła. U Ciebie gradient
jest rzędu 10⁻⁸, więc jego losowe wahania też są rzędu 10⁻⁸. Szum o wielkości 10⁻⁸ przesuwa wagi o 10⁻¹⁰ na krok, czyli nic.
Szum pomaga przy „płytkim” siodle, a nie przy gradiencie zredukowanym 10⁷ razy przez złą inicjalizację.
ADAM rozwiązał ten sam problem w 67 epok, bo zmienia **skalę** kroku, a nie tylko dokłada szum.

**Dziwny przypadek: koszt spadł do 0.6898 i utknął.** Skąd ten konkretny poziom? (rozdział 3.4)
- bias ostatniej warstwy ma gradient db = Σ (p − y)/N. **Nie ma w nim żadnego x** (wejścia warstwy), więc nie jest tłumiony przez
  zanikanie sygnału. To jedyny parametr, który ma „pełny” gradient;
- bias wyjścia uczy się więc normalnie i ustawia p na proporcję chorych: 111/242 = 0.459 (logit ln(0.459/0.541) = −0.165);
- koszt sieci, która każdemu mówi p = 0.459: −(0.459·ln 0.459 + 0.541·ln 0.541) = **0.6898**;
- wszystkie inne wagi dalej stoją. Sieć nauczyła się tylko „jaki odsetek pacjentów jest chory”, nic o cechach.
  Ponieważ 0.459 < 0.5, mówi „zdrowy” dla wszystkich, stąd F1 = 0.

**SGD szybciej dochodzi do tego fałszywego minimum** (około 10 epok), a potem koszt przestaje się zmieniać, więc warunek stopu
zadziałałby w 9. epoce. Paczki w 82., pełny zbiór nie zdążył w 300 epok. Im szybszy optymalizator, tym szybciej „wpada”
w fałszywą zbieżność.

### 14.7. Przeuczenie w części 2

Sieć z 5 warstwami ma 4089 parametrów na 242 pacjentów. Każdy dobrze działający wariant (He, ADAM, paczki, SGD) w końcu
zapamiętuje zbiór treningowy (koszt train 10⁻⁵), a koszt testowy rośnie. Najlepszy koszt testowy w części 2 (około 0.40–0.46)
jest **gorszy** niż w płytkiej sieci z części 1 (0.33). To nie jest problem optymalizacji, tylko zbyt dużego modelu dla tych danych.

**Dlaczego nie walczymy z tym (wczesne zatrzymanie, regularyzacja)?** Bo pytanie w części 2 dotyczy **optymalizacji**: jak szybko
i czy w ogóle optymalizator schodzi z kosztem treningowym. Przeuczenie jest skutkiem ubocznym tego, że optymalizacja działa
**dobrze**. W raporcie jest to opisane osobno, żeby nikt nie pomylił „wysokiego kosztu testowego” z „złą optymalizacją”.

### 14.8. Podsumowanie części 2 w pięciu zdaniach

1. Przy głębokiej sieci inicjalizacja decyduje, czy uczenie w ogóle ruszy: 0.01 zanika, 1 i 10 eksplodują, He zachowuje skalę sygnału.
2. ADAM ignoruje wielkość gradientu (krok ≈ lr), dzięki czemu wychodzi z płaskowyżu i nie rozbiega się, ale nie naprawia złej inicjalizacji.
3. Mniejsze paczki uczą się szybciej na epokę głównie dlatego, że robią więcej kroków; na krok różnica jest mała.
4. Szum z paczek nie wystarcza przy gradientach zredukowanych o 7 rzędów wielkości.
5. Warunek stopu „koszt przestał się zmieniać” jest zdradliwy na płaskowyżu; dlatego wszystkie przebiegi mają stały budżet epok.

---

## 15. Pytania prowadzącego i odpowiedzi

Odpowiedzi są krótkie, „do powiedzenia na głos”. Szczegóły są w rozdziałach podanych w nawiasach.

### Propagacja wsteczna i implementacja

**Co to jest propagacja wsteczna?**
Sposób liczenia gradientu kosztu po wszystkich wagach przez regułę łańcuchową. Najpierw forward od wejścia do kosztu z zapamiętaniem
wejść każdego modułu, potem backward od kosztu do wejścia. Każdy moduł dostaje ∂L/∂y, mnoży przez swoją lokalną pochodną i zwraca ∂L/∂x,
a moduł z wagami przy okazji zapisuje ∂L/∂W. (6.3)

**Dlaczego forward zapamiętuje wejście?**
Bo pochodne zależą od wartości z forward: dW = dᵀ·x wymaga x, a ReLU potrzebuje wiedzieć, które wejścia były dodatnie. (6.3, 8.4)

**Jakie wymiary ma dW i skąd wzór dᵀ·x?**
Taki sam jak W: (n_out, n_in). Waga W[j, i] wpływa na wyjście neuronu j każdego pacjenta z pochodną x[k, i], więc sumujemy
d[k, j]·x[k, i] po pacjentach, co macierzowo jest dᵀ·x. (6.4)

**Czym różni się Twoja konwencja od tej z PDF?**
W PDF x jest kolumną i warstwa liczy W·x. U mnie pacjenci są wierszami macierzy (N, 28), więc warstwa liczy X·Wᵀ + b. To ta sama
operacja w transpozycji, ale liczy od razu całą paczkę pacjentów jednym mnożeniem. W ma ten sam kształt (n_out, n_in) co w PDF. (5.4)

**Dlaczego backward ReLU to mnożenie element po elemencie, a nie macierz?**
Bo jakobian ReLU jest diagonalny (każde wyjście zależy tylko od swojego wejścia), a mnożenie przez macierz diagonalną to mnożenie
element po elemencie przez jej przekątną, czyli maskę [x > 0]. Dokładnie to uproszczenie opisuje PDF. (6.5)

**Dlaczego połączyłeś sigmoid z kosztem w jeden moduł?**
Matematycznie wynik jest ten sam: dL/dz = p − y. Ale osobno to się liczy jako (p − y)/(p(1 − p)) · p(1 − p), a dla dużego |z|
w float64 σ(z) to dokładnie 1.0 i p(1 − p) = 0, więc gradient by znikał, choć sieć myli się maksymalnie. Połączony moduł liczy od razu
p − y. Koszt liczę ze stabilnego wzoru max(z, 0) − zy + ln(1 + e^(−|z|)), który nigdy się nie przepełnia. (6.6, 6.7)

**Dlaczego dzielisz gradient przez N?**
Bo koszt to średnia po pacjentach, więc każdy pacjent wnosi 1/N. Dzięki temu gradient ma tę samą skalę niezależnie od rozmiaru paczki,
co jest ważne w części 2. (6.6)

**Skąd wiesz, że backward jest poprawny?**
Dwa testy. (1) Numeryczny: porównanie gradientu z backward z ilorazem różnicowym (L(w + ε) − L(w − ε))/2ε dla losowych wag w sieci
z dwiema warstwami ukrytymi; największy błąd względny 1.5·10⁻⁸. (2) Sieć bez warstw ukrytych daje F1 = 0.836, dokładnie jak regresja
logistyczna z lab2. (7, 9)

**Dlaczego w teście gradientu wagi mają std 0.5?**
Przy 0.01 gradienty głębszych warstw są tak małe, że różnica kosztów ginie w zaokrągleniach i test byłby niemiarodajny.
Wzory nie zależą od wartości wag, więc test sprawdza ten sam kod. (7.4)

**Dlaczego wagi są losowe, a nie zerowe?**
Przy równych wagach wszystkie neurony w warstwie liczą to samo i dostają ten sam gradient, więc zostają identyczne na zawsze.
Losowość łamie symetrię. Dodatkowo zerowe wagi to punkt siodłowy z zerowym gradientem. (8.4, 10.2)

### Wyniki części 1

**Dlaczego na początku uczenia jest plateau?**
Przy wagach 0.01 sieć startuje tuż przy punkcie siodłowym „wszystkie wagi = 0”. Gradient każdej warstwy jest proporcjonalny do wielkości
pozostałych warstw, więc jest malutki. Wagi rosną wykładniczo, ale z bardzo niskiego poziomu, więc przez długi czas nic nie widać,
a potem koszt gwałtownie spada. (10.2)

**Dlaczego większa warstwa szybciej wychodzi z plateau?**
Neuron wyjściowy sumuje h składników; suma rośnie jak √h, więc sygnał i gradient są silniejsze. (10.3)

**Dlaczego większa warstwa ma gorszy wynik na teście?**
Przeuczenie: więcej wag pozwala zapamiętać szum zbioru treningowego. Minimum kosztu testowego jest takie samo dla h ≥ 4,
ale większe sieci szybciej i mocniej się przeuczają po minimum. Różnica F1 to około 2 pacjentów, ale powtarzalna na 3 seedach. (10.3)

**Dlaczego bez normalizacji jest gorzej?**
Gradient wagi pierwszej warstwy jest proporcjonalny do wartości cechy. Cholesterol około 250 daje gradient 250 razy większy niż cecha 0/1,
a lr jest wspólny. Dla cholesterolu krok jest za duży (oscylacje), dla cech 0/1 za mały. Normalizacja daje wszystkim cechom podobną skalę. (10.4)

**Dlaczego sieć z 3 warstwami się nie nauczyła?**
Zanikanie sygnału i gradientu: każda warstwa mnoży skalę przez około 0.01·√(28/2) ≈ 0.037. Po 3 warstwach gradient jest tysiące razy
mniejszy i 10 000 epok nie wystarczy. Każda warstwa wydłuża plateau około 10–20 razy: 50 → 900 → 9000 epok. (10.5)

**Skąd √(28/2)?**
Wariancja sumy n = 28 składników waga × wejście to n·σ²·E[a²], a ReLU zabiera połowę wariancji (zeruje ujemne wartości). (10.5, 12.5)

**Dlaczego sieć wielowarstwowa nie jest lepsza od regresji logistycznej?**
Zależność w tych danych jest w przybliżeniu liniowa, a zbiór mały (242 pacjentów). Model liniowy wyciąga prawie całą informację,
a dodatkowe warstwy uczą się szumu. Zbiór nie jest liniowo separowalny (koszt regresji logistycznej zatrzymuje się na 0.31),
ale reszta błędów to szum, którego żaden model z tych cech nie przewidzi. (10.6)

**Dlaczego nie wybierasz modelu z najlepszej epoki?**
Najlepsza epoka według zbioru testowego to podglądanie testu. Uczciwie wymaga to osobnego zbioru walidacyjnego, co zmniejszyłoby
zbiór treningowy i zepsuło porównanie z lab2. Minimum kosztu testowego podaję tylko informacyjnie. (1.2 raportu)

**Co znaczy „±” i dlaczego 3 seedy?**
Średnia ± odchylenie standardowe z 3 przebiegów z różnymi wagami początkowymi (podział danych stały). Przy 61 pacjentach testowych
jeden pacjent to około 0.02 F1, więc bez powtórzeń nie da się odróżnić efektu od przypadku. 3 to kompromis między wiarygodnością
a czasem obliczeń. Uwaga: to odchylenie populacyjne (dzielenie przez 3); odchylenie z próby (przez 2) byłoby około 1.22 razy większe.
Przy 3 próbkach to tylko orientacyjna miara rozrzutu. (10.7, 8.11)

### Część 2: optymalizacja

**Co to jest punkt siodłowy i gdzie go widać w Twoich eksperymentach?**
Miejsce z zerowym gradientem, które nie jest minimum: w jednym kierunku funkcja rośnie, w innym maleje (np. x² − y² w (0, 0)).
U mnie: punkt „wszystkie wagi = 0”; inicjalizacja 0.01 startuje tuż przy nim, stąd plateau. (11.2)

**Dlaczego w głębokich sieciach siodła są ważniejsze niż minima lokalne?**
Żeby punkt z zerowym gradientem był minimum, funkcja musi rosnąć we wszystkich tysiącach kierunków naraz. Prawie zawsze w którymś
maleje, więc prawie każdy taki punkt jest siodłem. (11.2)

**Jak działa ADAM?**
Dla każdej wagi trzyma średnią kroczącą gradientu m (momentum, kierunek) i średnią kroczącą kwadratu gradientu v (wielkość).
Krok to lr · m̂/(√v̂ + ε). Dzielenie przez √v̂ normalizuje krok: niezależnie od wielkości gradientu krok ma długość około lr. (12.4)

**Po co korekcja biasu?**
m i v startują od 0, więc na początku są zaniżone (po pierwszym kroku m = 0.1·g). Dzielenie przez 1 − βᵗ to naprawia (m̂ = g po pierwszym kroku).
Z czasem βᵗ → 0 i korekcja znika. (12.4)

**Czy wzory ADAM w PDF są poprawne?**
Są w nich literówki: nawias w m powinien obejmować tylko (1 − β1), a w v powinien być kwadrat gradientu. Bez kwadratu v mogłoby być
ujemne i nie mierzyłoby wielkości gradientu. Zaimplementowałem wersję z oryginalnej pracy (Kingma & Ba). (12.4)

**Po co ε w ADAM?**
Żeby nie dzielić przez 0. Ciekawostka: przy gradientach 10⁻⁸ (inicjalizacja 0.01) ε = 10⁻⁸ jest tego samego rzędu, więc krok
to około lr/2, nadal miliony razy więcej niż w GD. (12.4)

**Dlaczego ADAM ma inny lr niż GD? Czy to uczciwe porównanie?**
0.001 to standardowa wartość dla ADAM, 0.01 dla GD jest z treści. ADAM ma więc 10 razy **mniejszy** lr, a i tak wygrywa, więc jego
przewaga nie wynika z większego kroku. Lr nie jest wprost porównywalny, bo w GD krok = lr·gradient, a w ADAM krok ≈ lr.

**Dlaczego ADAM wychodzi z płaskowyżu, a GD nie?**
GD robi krok lr·10⁻⁸ = 10⁻¹⁰. ADAM dzieli gradient przez jego własną wielkość, więc robi krok około 0.001. Przez kilkadziesiąt epok
powiększa wagi, sygnał przestaje zanikać, gradient rośnie z 10⁻⁸ do 10⁻² i sieć zaczyna się uczyć (67 epok). (14.4)

**Czy ADAM zawsze jest lepszy?**
Nie. Przy inicjalizacji N(0, 1) nie rozbiega się (GD tak, w 1/3 przebiegów), ale uczy się wolniej (krok ≈ 0.001 przy wagach rzędu 1)
i kończy z bardzo pewnym siebie, przeuczonym modelem (koszt testowy 75). Szybciej też przechodzi w przeuczenie. Nie naprawia złej
inicjalizacji. (14.4)

**Skąd koszt testowy 75 przy F1 0.66?**
Duże wagi = ogromne logity. Każda pomyłka na teście jest „pewna na 100%” (np. logit −200 dla chorego daje koszt około 200),
a koszt nie jest ograniczony z góry. Średnia z 61 pacjentów to około 75. F1 patrzy tylko na decyzje 0/1, więc jest „normalne”. (14.4)

**Skąd wzór He √(2/n)?**
Wariancja wyjścia warstwy = n·σ²·½·wariancja wejścia (½ przez ReLU). Żeby skala się nie zmieniała, n·σ²/2 = 1, czyli σ = √(2/n).
Dla 28 wejść: 0.267. (12.5)

**Dlaczego N(0, 1) jest złą inicjalizacją przy 28 wejściach?**
Czynnik na warstwę to 1·√14 ≈ 3.7, więc po 5 warstwach sygnał rośnie około 700 razy. Logity są ogromne, koszt startowy około 750,
gradienty około 10³ i kroki GD za duże: koszt rośnie, a w 1/3 przebiegów uczenie się rozbiega. (14.3)

**Co to jest „zepsuta” inicjalizacja u Ciebie?**
N(0, 10): czynnik 37 na warstwę, po 5 warstwach 7·10⁷. Wszystkie przebiegi rozbiegają się w 1. epoce. Dodatkowo pokazuję N(0, 0.01)
jako zepsutą w drugą stronę (zanikanie). (14.3)

**Dlaczego zepsuta inicjalizacja ma F1 0.42?**
Bo dwa z trzech rozbieżnych modeli mówią „chory” dla wszystkich, a wtedy F1 = 2·28/(61 + 28) = 0.629. F1 jest mylące dla takich
modeli, dlatego patrzę na koszt i rozbieżność. (14.3)

**Czym różnią się tryby uczenia?**
Pełny zbiór: 1 dokładny krok na epokę. Paczki po 32: 8 przybliżonych kroków. SGD: 242 bardzo zaszumione kroki. Na epokę mniejsze
paczki uczą się dużo szybciej, bo robią więcej kroków; na krok różnica jest mała. SGD jest najwolniejszy w czasie i najbardziej
zaszumiony. Paczki to najlepszy kompromis. (12.1, 14.5)

**Dlaczego SGD wolniej liczy się w czasie, skoro uczy się szybciej?**
242 małe mnożenia macierzy w pętli Pythona kosztują dużo więcej niż jedno duże (narzut wywołań; numpy jest szybki na dużych macierzach):
około 20× wolniej na epokę. (14.5)

**Czy szum SGD pomógł wyjść z płaskowyżu?**
Nie. Przy gradiencie 10⁻⁸ szum też jest rzędu 10⁻⁸ i nie przesuwa wag. Treść mówi, że może „odrobinę” pomóc; u mnie problem był za duży.
ADAM pomógł, bo zmienia skalę kroku. (14.6)

**Skąd koszt 0.6898?**
Bias wyjścia ma pełny gradient (bez x, więc bez zanikania) i ustawia p na proporcję chorych 0.459. Koszt takiej sieci to entropia
proporcji klas, 0.6898. Reszta wag stoi, sieć mówi „zdrowy” wszystkim. (3.4, 14.6)

**Dlaczego nie użyłeś warunku stopu z lab2?**
Bo na płaskowyżu koszt prawie się nie zmienia i warunek uznałby utknięcie za zbieżność: GD z inicjalizacją 0.01 zatrzymałby się
w epoce 569 z kosztem 0.69, SGD w epoce 9. Dlatego stały budżet epok, a moment zatrzymania warunku podaję w tabeli jako ostrzeżenie. (11.4)

**Jak wykrywasz rozbieżność?**
Koszt nieskończony albo ponad 10⁴ razy większy niż przed uczeniem. Sam test na NaN nie wystarcza, bo stabilny wzór kosztu daje skończone
liczby nawet 10²⁸⁷. Próg względny, bo przy N(0, 10) już koszt startowy to 7·10⁸. Koszty rozbieżnych przebiegów pomijam w średnich. (13.4, 13.5)

**Dlaczego 5 warstw w części 2?**
Treść każe dobrać głębokość tak, żeby problemy było widać. Efekt inicjalizacji jest wykładniczy w liczbie warstw: przy 1–2 warstwach
wszystkie inicjalizacje działałyby podobnie, przy 5 różnice są ogromne. W części 1 już 3 warstwy z 0.01 się nie uczyły. (11.3)

**Dlaczego bias = 0 w części 2, a w części 1 losowy?**
W części 1 treść kazała inicjalizować wagi z N(0, 0.01), więc bias też. W części 2 porównuję rozkłady wag, a bias nie wpływa na wariancję
sygnału; zerowy bias to standard przy He. Symetrię łamią losowe wagi. (12.5)

**Dlaczego część 2 importuje z części 1, zamiast mieć własny kod?**
Bo moduły sieci przeszły test gradientu. Część 2 używa dokładnie tego samego, sprawdzonego backwardu i tych samych danych.
Nowy kod to tylko inicjalizacja, optymalizatory i pętla z paczkami. (13.1)

### Pytania „podchwytliwe” i słabe strony (warto znać, zanim ktoś zapyta)

- **Tylko jeden podział train/test.** Wyniki dotyczą tego jednego podziału (ten sam co w lab2). Walidacja krzyżowa dałaby pewniejsze
  wnioski o F1, ale celem było porównanie zachowania uczenia, nie dokładny pomiar F1.
- **3 seedy to mało.** Wystarczy, żeby odróżnić duże efekty od przypadku (np. h = 512 0.793, 0.814, 0.793), ale ± jest tylko orientacyjne.
- **lr nie był strojony.** GD ma lr z treści (0.01), ADAM standardowe 0.001. Inne lr dałyby inne liczby, ale nie zmieniłyby kierunku wniosków.
- **W trybach uczenia jest tylko jeden rozmiar paczki (32).** Więcej rozmiarów pokazałoby trend, ale treść tego nie wymaga.
- **F1 na teście w części 2 jest niższe niż w części 1.** To przeuczenie dużej sieci, nie błąd optymalizacji (14.7).
- **W części 1 koszt train w historii jest liczony przed krokiem, a test po kroku** (przesunięcie o jedną epokę). Przy 10 000 epok bez znaczenia.

---

## 16. Słowniczek i ściąga

### 16.1. Słowniczek

| Pojęcie | Znaczenie |
|---|---|
| cecha (feature) | jedna liczba opisująca pacjenta (wiek, cholesterol, cp_4…) |
| one-hot | zamiana kategorii na kolumny 0/1, po jednej na kategorię |
| normalizacja (standaryzacja) | (x − średnia)/odchylenie, cecha ma średnią 0 i odchylenie 1 |
| zbiór treningowy / testowy | na czym sieć się uczy / na czym sprawdzamy ją na nowych danych |
| waga, bias | parametry neuronu: mnożnik każdego wejścia, stała |
| logit z | ważona suma przed sigmoidem, dowolna liczba |
| sigmoid σ | 1/(1 + e^(−z)), zamienia logit na prawdopodobieństwo (0, 1) |
| ReLU | max(0, z), aktywacja w warstwach ukrytych |
| warstwa ukryta | warstwa między wejściem a wyjściem |
| MLP | sieć wielowarstwowa: [Linear → ReLU] × k → Linear |
| koszt (loss), BCE | entropia krzyżowa −[y ln p + (1−y) ln(1−p)], średnia po pacjentach |
| gradient | wektor pochodnych kosztu po wszystkich wagach, „kierunek pod górę” |
| spadek gradientu (GD) | w ← w − lr·gradient |
| lr | learning rate, długość kroku |
| epoka | jedno przejście przez cały zbiór treningowy |
| krok (iteracja) | jedna aktualizacja wag |
| paczka (mini-batch) | podzbiór pacjentów, z którego liczymy jeden krok |
| SGD | krok z jednego pacjenta (paczka rozmiaru 1) |
| forward / backward | przejście od wejścia do kosztu / od kosztu wstecz z gradientami |
| propagacja wsteczna | liczenie gradientów regułą łańcuchową od końca sieci |
| plateau, płaskowyż | obszar, gdzie gradient jest prawie zerowy i koszt stoi |
| punkt siodłowy | gradient = 0, ale nie minimum (w jednym kierunku w górę, w innym w dół) |
| zanikające / eksplodujące gradienty | gradient maleje / rośnie wykładniczo z liczbą warstw |
| inicjalizacja He | wagi z N(0, √(2/n_in)), zachowuje skalę sygnału przy ReLU |
| momentum | średnia krocząca gradientów, „rozpęd” |
| ADAM | momentum + skalowanie kroku przez średnią kwadratu gradientu, krok ≈ lr |
| korekcja biasu | dzielenie m, v przez 1 − βᵗ, bo startują od zera |
| przeuczenie (overfitting) | koszt train spada, koszt test rośnie: model zapamiętuje szum |
| rozbieżność | koszt rośnie bez końca, uczenie „wybucha” |
| seed | ziarno losowania, ten sam seed = te same liczby losowe |
| TP / FP / FN / TN | trafiony chory / fałszywy alarm / przeoczony chory / trafiony zdrowy |
| accuracy / precision / recall / F1 | odsetek trafień / trafność alarmów / wykrywalność chorych / ich średnia harmoniczna |

### 16.2. Najważniejsze wzory

```
Linear forward:     Z = X·Wᵀ + b
Linear backward:    dW = dᵀ·X,   db = Σ_wiersze d,   dX = d·W
ReLU backward:      dX = d ⊙ [X > 0]
Sigmoid+BCE:        L = mean( max(z,0) − z·y + ln(1 + e^(−|z|)) ),   dL/dz = (p − y)/N
GD:                 w ← w − lr·g
ADAM:               m = β1·m + (1−β1)·g,   v = β2·v + (1−β2)·g²
                    m̂ = m/(1−β1ᵗ),   v̂ = v/(1−β2ᵗ),   w ← w − lr·m̂/(√v̂ + ε)
Wariancja warstwy:  Var(z_l) = n·σ²·½·Var(z_{l−1})   →   czynnik σ·√(n/2) na warstwę
He:                 σ = √(2/n_in)
Gradient check:     (L(w+ε) − L(w−ε))/(2ε),   błąd = |a − b|/(|a| + |b|)
F1:                 2TP/(2TP + FP + FN)
```

### 16.3. Ściąga: najważniejsze liczby

| Co | Liczba |
|---|---|
| dane | 303 pacjentów → 242 train / 61 test (28 chorych), 28 cech |
| koszt zgadywania | ln 2 = 0.693 |
| koszt „zna tylko proporcję klas” | 0.6898 |
| błąd sprawdzenia gradientu | 1.5·10⁻⁸ |
| regresja logistyczna (0 warstw) | F1 0.836 (jak lab2) |
| plateau: 0 / 1 / 2 / 3 warstwy | 48 / 907 / 9031 / nie wyszła |
| plateau: h = 2 / 28 / 512 | 1882 / 907 / 402 |
| F1: znormalizowane vs surowe | 0.832 vs 0.658 |
| skala sygnału na warstwę: 0.01 / He / 1 / 10 | ×0.037 / ×1 / ×3.7 / ×37 |
| He dla 28 wejść | σ = 0.267 |
| 5 warstw: plateau GD He / ADAM He | 107 / 15 epok |
| 5 warstw, init 0.01: GD / ADAM | nie wychodzi / 67 epok |
| N(0, 1) GD: rozbieżność | 1/3 przebiegów |
| N(0, 10) GD: rozbieżność | 3/3 w 1. epoce |
| warunek stopu na plateau (GD, 0.01) | zatrzymałby w epoce 569 z kosztem 0.69 |
| tryby (He): plateau pełny / 32 / SGD | 107 / 14 / 1 epok |
| czas przebiegu pełny / 32 / SGD (300 epok) | 0.3 / 0.5 / 6.3 s |

### 16.4. Cała historia ćwiczenia w 10 zdaniach

1. Zbudowałem sieć z modułów, z których każdy umie forward (policz wyjście, zapamiętaj wejście) i backward (oddaj gradient, zapisz dW).
2. Backward to reguła łańcuchowa; poprawność sprawdziłem numerycznie (błąd 10⁻⁸) i na regresji logistycznej (F1 jak w lab2).
3. Przy wagach 0.01 sieć startuje przy punkcie siodłowym, stąd plateau; większa warstwa wychodzi szybciej, ale mocniej się przeucza.
4. Bez normalizacji duże cechy dają ogromne gradienty, a wspólny lr nie pasuje do wszystkich wag, więc uczenie jest niestabilne.
5. Każda warstwa przy 0.01 osłabia sygnał około 27 razy, więc 3 warstwy już się nie uczą; MLP i tak nie jest lepsze od regresji logistycznej, bo dane są prawie liniowe.
6. W części 2 przy 5 warstwach: 0.01 zanika, N(0, 1) i N(0, 10) eksplodują, He zachowuje skalę, bo √(2/n) równoważy sumę n wejść i połowę zabraną przez ReLU.
7. ADAM normalizuje krok do około lr, więc wychodzi z płaskowyżu (67 epok zamiast nigdy) i nie rozbiega się, ale nie naprawia złej inicjalizacji.
8. Mniejsze paczki uczą się szybciej na epokę, bo robią więcej kroków; na krok różnica jest mała, a paczki są najlepszym kompromisem czasu i jakości.
9. Szum z paczek nie wyciągnął sieci z płaskowyżu, a warunek stopu „koszt się nie zmienia” uznałby utknięcie za zbieżność.
10. Dobre optymalizatory przeuczają dużą sieć na małym zbiorze; to osobny problem od optymalizacji.

---

## Załącznik: pełne listingi kodu

Poniżej trzy pliki w całości, dokładnie w takiej postaci, w jakiej są w folderze `lab3`. Wyjaśnienia każdego fragmentu są w rozdziałach 8, 9 i 13.

### `lab3.py`

```python
import os
# macierze są małe (242 x 28 ... 512), więc wielowątkowe mnożenie macierzy tylko spowalnia
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STD_INIT = 0.01      # wagi z N(0, 0.01) - z treści zadania
LR = 0.01            # współczynnik uczenia - z treści zadania
EPOCHS = 10000
SEEDS = [0, 1, 2]    # każdą konfigurację uczymy od 3 różnych losowań wag początkowych
PLATEAU_LOSS = 0.6   # "wyjście z plateau" = pierwsza epoka z kosztem train < 0.6 (start: ln 2 ≈ 0.693)

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

# podział stały (ten sam random_state co w lab2) - seedy zmieniają tylko wagi początkowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=67)

# dane nieznormalizowane (do eksperymentu 2)
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
# X ma kształt (liczba próbek, liczba cech) - każdy wiersz to jeden pacjent.
# forward(x)  - liczy wyjście i zapamiętuje wejście x
# backward(d) - dostaje d = dL/dy (jak koszt zależy od wyjścia modułu),
#               zwraca dL/dx (jak koszt zależy od wejścia modułu)

class Linear:
    # y = W x + b
    def __init__(self, n_in, n_out, rng, std=STD_INIT):
        self.W = rng.normal(0, std, (n_out, n_in))
        self.b = rng.normal(0, std, (1, n_out))

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


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class SigmoidBCELoss:
    # sigmoid + koszt jak w regresji logistycznej (lab2), połączone w jeden moduł.
    # Wejściem jest logit z (wyjście ostatniej warstwy Linear), a nie prawdopodobieństwo p.
    def forward(self, z, y):
        self.p = sigmoid(z)
        self.y = y.reshape(-1, 1)
        # -y ln p - (1-y) ln(1-p) zapisane stabilnie: max(z,0) - z y + ln(1 + e^-|z|)
        return np.mean(np.maximum(z, 0) - z * self.y + np.log1p(np.exp(-np.abs(z))))

    def backward(self):
        # dL/dz = dL/dp * dp/dz = (p - y) / (p(1-p)) * p(1-p) = p - y   (/ N, bo koszt to średnia)
        return (self.p - self.y) / len(self.y)


class MLP:
    # [Linear -> ReLU] x liczba warstw ukrytych -> Linear  (sigmoid jest w funkcji kosztu)
    def __init__(self, sizes, seed=0, std=STD_INIT):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng, std))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:              # od pierwszej warstwy do ostatniej
            x = layer.forward(x)
        return x                               # logit z

    def predict_proba(self, x):
        return sigmoid(self.forward(x)).ravel()

    def backward(self, d):
        for layer in reversed(self.layers):    # od ostatniej warstwy do pierwszej
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
    # hidden - lista rozmiarów warstw ukrytych, np. [28] albo [28, 28]
    model = MLP([n_features] + hidden + [1], seed=seed)
    loss_fn = SigmoidBCELoss()
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

    return model, {k: np.array(v) for k, v in history.items()}


def mean_std(values):
    # średnia i odchylenie z pominięciem nan (epoka wyjścia z plateau, gdy sieć nie wyszła)
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def experiment(configs, title, filename):
    # configs: {nazwa: (warstwy ukryte, X_train, X_test)}
    # każda konfiguracja uczona dla wszystkich SEEDS, wynik = średnia ± odchylenie standardowe
    print(f"\n=== {title} ===  (średnia ± odchylenie z {len(SEEDS)} seedów)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_axis = np.arange(1, EPOCHS + 1)   # od 1, bo na skali log punkt x = 0 by zniknął
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
    # h = liczba neuronów w warstwie ukrytej; liczba cech = 28
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
```

### `sprawdzenie_gradientu.py`

```python
# Osobny, szybki test (< 1 s): czy backward() liczy pochodne poprawnie.
# Dla losowych wag porównujemy dL/dW z backward z ilorazem różnicowym
#   (L(w + eps) - L(w - eps)) / (2 eps)
# Moduły i dane bierzemy z lab3.py (import nie uruchamia eksperymentów).
import numpy as np
from lab3 import MLP, Linear, SigmoidBCELoss, X_train, y_train, n_features

EPS = 1e-6
N_CHECKS = 20     # ile losowych wag sprawdzamy w każdej macierzy W i wektorze b

# Mała sieć z 2 warstwami ukrytymi (sprawdza też ReLU). Wagi std = 0.5 zamiast 0.01,
# żeby gradienty nie były bliskie zeru - wtedy błąd względny jest miarodajny.
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
```

### `lab3_cz2.py`

```python
# Ćwiczenie 3, część 2: ADAM, tryby uczenia (pełny zbiór / paczki / pojedyncze przykłady), inicjalizacja wag.
# Moduły sieci i dane bierzemy z lab3.py (sprawdzone testem gradientu) - import nie uruchamia eksperymentów części 1.
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from lab3 import (MLP, Linear, SigmoidBCELoss, X_train, X_test, y_train, y_test, n_features,
                  metrics, mean_std)

SEEDS = [0, 1, 2]
PLATEAU_LOSS = 0.6   # "koniec plateau" = pierwsza epoka z kosztem train < 0.6 (start: ln 2 ≈ 0.693), jak w części 1
TOL = 1e-6           # klasyczny warunek stopu z lab2: |L_k - L_(k-1)| < TOL
DIVERGE_FACTOR = 1e4 # rozbieżność = koszt train > 10^4 x koszt przed uczeniem (albo nan / inf)
HIDDEN = [n_features] * 5    # 5 warstw ukrytych po 28 neuronów (sieć "głęboka" dla tego zbioru)
LR_GD = 0.01                 # jak w części 1
LR_ADAM = 0.001              # standardowa wartość dla ADAM

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean po rozbieżnych przebiegach (same nan)


# ---------------------------------------------------------------
# 1. INICJALIZACJA WAG
# ---------------------------------------------------------------
def linear_layers(model):
    return [layer for layer in model.layers if isinstance(layer, Linear)]


def init_weights(model, init, seed):
    # init: "he" -> N(0, sqrt(2 / d_in)), osobne odchylenie dla każdej warstwy
    #       liczba -> N(0, init) dla wszystkich warstw (np. 0.01, 1.0, 10.0)
    # bias = 0, żeby porównanie dotyczyło tylko rozkładu wag W
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
    # m - średnia krocząca gradientu (momentum), v - średnia krocząca kwadratu gradientu.
    # Oba osobno dla każdej wagi, startują od zera, stąd korekcja biasu przez 1 - beta^t.
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
            # rozbieżność: koszt nieskończony albo DIVERGE_FACTOR razy większy niż przed uczeniem.
            # Sam test na nan nie wystarcza - stabilny wzór kosztu daje skończone, ale absurdalne wartości (np. 1e287)
            loss = h["train_loss"][-1]
            if not np.isfinite(loss) or loss > DIVERGE_FACTOR * max(h["train_loss"][0], 1.0):
                h["diverged_at"] = epoch
                break

    # po rozbieżności dopełniamy historię wartością nan, żeby wszystkie przebiegi miały tę samą długość
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
    if h["diverged_at"]:
        # koszt rozbieżnego przebiegu (np. 1e11) zniekształciłby średnią pozostałych seedów -
        # pomijamy go w średniej, liczba rozbieżnych przebiegów jest wypisywana osobno
        for k in ["train_loss", "test_loss", "min_test_loss"]:
            m[k] = np.nan
    return m


# ---------------------------------------------------------------
# 4. EKSPERYMENT - każda konfiguracja dla 3 seedów, tabela + wykresy (średnia ± odchylenie)
# ---------------------------------------------------------------
def fmt(agg, k, p=".3g"):
    mean, std = agg[k]
    return "—" if np.isnan(mean) else f"{mean:{p}} ± {std:{p}}"


def experiment(configs, title, filename, epochs, by_steps=False):
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
        print(f"{'':24s} koszt train = {fmt(agg, 'train_loss')}  koszt test = {fmt(agg, 'test_loss')}  "
              f"min koszt test = {fmt(agg, 'min_test_loss')}  accuracy = {fmt(agg, 'accuracy', '.3f')}  "
              f"F1 = {fmt(agg, 'f1', '.3f')}  [F1 dla seedów: {', '.join(f'{m['f1']:.3f}' for m in ms)}]")

    plot_experiment(results, title, filename, epochs, by_steps)
    return results


def band(ax, x, curves, label, log=False):
    curves = np.where(np.isfinite(curves) & (np.abs(curves) < 1e30), curves, np.nan)   # bez inf na wykresie
    mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
    line, = ax.plot(x, mean, label=label)
    low = np.maximum(mean - std, mean * 0.1) if log else mean - std    # na skali log pas nie może zejść do <= 0
    ax.fill_between(x, low, mean + std, color=line.get_color(), alpha=0.2)


def plot_experiment(results, title, filename, epochs, by_steps):
    panels = [("train_loss", "koszt (train)"), ("test_loss", "koszt (test)"), ("f1", "F1 (test)"),
              ("grad1", "|dW| warstwy 1 (pełny gradient)")]
    if by_steps:
        panels.insert(1, ("train_loss_steps", "koszt (train)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
    x = np.arange(epochs + 1)
    for name, (agg, hs) in results.items():
        for ax, (key, _) in zip(axes, panels):
            if key == "grad1":
                band(ax, x, np.stack([h["grad_norms"][:, 0] for h in hs]), name, log=True)
            elif key == "train_loss_steps":
                band(ax, hs[0]["steps"], np.stack([h["train_loss"] for h in hs]), name, log=True)
            else:
                band(ax, x, np.stack([h[key] for h in hs]), name, log=key != "f1")
    for ax, (key, ylabel) in zip(axes, panels):
        ax.set_xscale("symlog", linthresh=1)          # symlog: oś log, ale epoka 0 (przed uczeniem) jest widoczna
        ax.set_xlabel("liczba kroków uczenia (symlog)" if key == "train_loss_steps" else "epoka (symlog)")
        ax.set_ylabel(ylabel)
        if key != "f1":
            ax.set_yscale("log")
        ax.grid(which="both", alpha=0.4)
        ax.legend(fontsize=8)
    plt.suptitle(f"{title}\nlinia = średnia z {len(SEEDS)} seedów, pas = ± odchylenie standardowe; "
                 f"oś Y kosztu w skali log")
    plt.tight_layout()
    plt.savefig(f"wykresy/{filename}")
    plt.close()


def plot_init_layers(results, filename):
    # odchylenie wyjścia i norma gradientu każdej warstwy PRZED uczeniem (średnia z seedów)
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
# 5. TEST ADAM (szybki, sprawdza implementację na prostej funkcji)
# ---------------------------------------------------------------
def test_adam():
    # f(w) = (w - 3)^2, start w = 5. Dwie własności ADAM:
    #  - pierwszy krok ma długość lr niezależnie od wielkości gradientu (m_hat / sqrt(v_hat) = g / |g| = ±1)
    #  - po wielu krokach w -> 3
    print("=== Test ADAM na f(w) = (w - 3)^2, start w = 5, lr = 0.01 ===")
    for scale in [1.0, 1e6]:                          # ten sam problem, gradient przemnożony przez 10^6
        opt = Adam(lr=0.01)
        w = np.array([5.0])
        for t in range(1, 3001):
            opt.t = t
            opt.update("w", w, scale * 2 * (w - 3))
            if t == 1:
                first_step = 5.0 - w[0]
        print(f"skala gradientu {scale:6.0e}: pierwszy krok = {first_step:.5f}, w po 3000 krokach = {w[0]:.4f}")


# ---------------------------------------------------------------
# 6. EKSPERYMENTY
# ---------------------------------------------------------------
def main(scale=1.0):
    # scale < 1 - szybka próba ze zmniejszoną liczbą epok (sprawdzenie, czy wszystko działa, i pomiar czasu)
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
        "Inicjalizacja wag (5 warstw ukrytych, GD lr = 0.01, pełny zbiór)", "cz2_inicjalizacja.png", ep(2000),
    )
    plot_init_layers(res_init, "cz2_inicjalizacja_warstwy.png")

    # EKSPERYMENT 2 - GD vs ADAM (pełny zbiór)
    experiment(
        {
            "GD, init 0.01": (HIDDEN, 0.01, GD, None),
            "ADAM, init 0.01": (HIDDEN, 0.01, Adam, None),
            "GD, init He": (HIDDEN, "he", GD, None),
            "ADAM, init He": (HIDDEN, "he", Adam, None),
            "GD, init 1": (HIDDEN, 1.0, GD, None),
            "ADAM, init 1": (HIDDEN, 1.0, Adam, None),
        },
        f"GD (lr = {LR_GD}) vs ADAM (lr = {LR_ADAM}), 5 warstw ukrytych, pełny zbiór", "cz2_gd_vs_adam.png", ep(2000),
    )

    # EKSPERYMENT 3 - TRYBY UCZENIA (zwykły GD), osobno dla inicjalizacji He i 0.01 (plateau)
    for init, label, fname in [("he", "He", "cz2_tryby_he.png"), (0.01, "N(0, 0.01)", "cz2_tryby_001.png")]:
        experiment(
            {
                "pełny zbiór (242)": (HIDDEN, init, GD, None),
                "paczki po 32": (HIDDEN, init, GD, 32),
                "pojedyncze przykłady (1)": (HIDDEN, init, GD, 1),
            },
            f"Tryby uczenia, init {label} (5 warstw ukrytych, GD lr = {LR_GD})", fname, ep(300), by_steps=True,
        )
    print(f"\nCałkowity czas: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
```
