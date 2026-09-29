import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Ustawienia dla lepszych wykresów
plt.style.use("default")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)


# Macierz korelacji dla zmiennych numerycznych
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

print("Macierz korelacji:")
print(correlation_matrix.round(3))

# Heatmapa korelacji
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")

plt.title("Heatmapa korelacji cech numerycznych")
plt.show()
