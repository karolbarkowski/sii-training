import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# Ustawienia dla lepszych wykresów
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------------
# Zadanie 2.1: Podstawowe histogramy
# Subplot z 4 histogramami (2x2) dla: powierzchnia, cena, wiek_domu, odleglosc_centrum
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram powierzchni
axes[0, 0].hist(
    df["powierzchnia"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
)
axes[0, 0].set_title("Rozkład powierzchni domów")
axes[0, 0].set_xlabel("Powierzchnia (m²)")
axes[0, 0].set_ylabel("Liczba domów")

# Histogram cen
axes[0, 1].hist(df["cena"], bins=30, alpha=0.7, color="lightgreen", edgecolor="black")
axes[0, 1].set_title("Rozkład cen domów")
axes[0, 1].set_xlabel("Cena (PLN)")
axes[0, 1].set_ylabel("Liczba domów")


# Histogram wieku domów
axes[1, 0].hist(df["wiek_domu"], bins=30, alpha=0.7, color="salmon", edgecolor="black")
axes[1, 0].set_title("Rozkład wieku domów")
axes[1, 0].set_xlabel("Wiek domu (lata)")
axes[1, 0].set_ylabel("Liczba domów")


# Histogram odległości od centrum
axes[1, 1].hist(
    df["odleglosc_centrum"], bins=30, alpha=0.7, color="orchid", edgecolor="black"
)
axes[1, 1].set_title("Rozkład odległości od centrum")
axes[1, 1].set_xlabel("Odległość od centrum (km)")
axes[1, 1].set_ylabel("Liczba domów")


plt.tight_layout()
plt.show()
