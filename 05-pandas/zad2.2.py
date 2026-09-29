from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ustawienia dla lepszych wykresów
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------------
# Zadanie 2.2: Scatter plots - szukamy zależności!
# Subplot z 3 scatter plotami (1x3)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatter plot: powierzchnia vs cena
axes[0].scatter(df["powierzchnia"], df["cena"], alpha=0.6, color="blue")
axes[0].set_xlabel("Powierzchnia (m²)")
axes[0].set_ylabel("Cena (PLN)")
axes[0].set_title("Powierzchnia vs Cena")

# Scatter plot: wiek_domu vs cena
axes[1].scatter(df["wiek_domu"], df["cena"], alpha=0.6, color="green")
axes[1].set_xlabel("Wiek domu (lata)")
axes[1].set_ylabel("Cena (PLN)")
axes[1].set_title("Wiek domu vs Cena")


# Scatter plot: odleglosc_centrum vs cena
axes[2].scatter(df["odleglosc_centrum"], df["cena"], alpha=0.6, color="red")
axes[2].set_xlabel("Odległość od centrum (km)")
axes[2].set_ylabel("Cena (PLN)")
axes[2].set_title("Odległość od centrum vs Cena")


plt.tight_layout()
plt.show()
