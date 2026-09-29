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
# Zadanie 2.3: Box plots - porównania między kategoriami
# Subplot z 4 box plotami (2x2)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Box plot: cena według dzielnic
# Stwórz box plot pokazujący ceny w różnych dzielnicach
axes[0, 0].boxplot(
    [df[df["dzielnica"] == dz]["cena"] for dz in df["dzielnica"].unique()],
)
axes[0, 0].set_xticklabels(df["dzielnica"].unique().tolist())
axes[0, 0].set_title("Ceny według dzielnic")
axes[0, 0].set_xlabel("Dzielnica")
axes[0, 0].set_ylabel("Cena (PLN)")


# Box plot: cena według liczby pokoi
axes[0, 1].boxplot(
    [df[df["liczba_pokoi"] == lp]["cena"] for lp in df["liczba_pokoi"].unique()],
)
axes[0, 1].set_xticklabels(df["liczba_pokoi"].unique().tolist())
axes[0, 1].set_title("Ceny według liczby pokoi")
axes[0, 1].set_xlabel("Liczba pokoi")
axes[0, 1].set_ylabel("Cena (PLN)")

# Box plot: czy domy z garażem są droższe?
axes[1, 0].boxplot(
    [df[df["garaz"] == g]["cena"] for g in df["garaz"].unique()],
)
axes[1, 0].set_xticklabels(df["garaz"].unique().tolist())
axes[1, 0].set_title("Ceny domów z/bez garażu")
axes[1, 0].set_xlabel("Garaż")
axes[1, 0].set_ylabel("Cena (PLN)")

# Box plot: czy domy z balkonem są droższe?
axes[1, 1].boxplot(
    [df[df["balkon"] == b]["cena"] for b in df["balkon"].unique()],
)
axes[1, 1].set_xticklabels(df["balkon"].unique().tolist())
axes[1, 1].set_title("Ceny domów z/bez balkonu")
axes[1, 1].set_xlabel("Balkon")
axes[1, 1].set_ylabel("Cena (PLN)")


plt.tight_layout()
plt.show()
