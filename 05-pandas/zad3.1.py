from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)

# ZADANIE 3.1: Identyfikacja outlierów

# TODO: Znajdź outliers w cenie używając metody IQR (Inter-Quartile Range)
Q1 = df["cena"].quantile(0.25)
Q3 = df["cena"].quantile(0.75)
IQR = Q3 - Q1

# Granice outlierów
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 (25% kwantyl): {Q1:,.0f} PLN")
print(f"Q3 (75% kwantyl): {Q3:,.0f} PLN")
print(f"IQR: {IQR:,.0f} PLN")
print(f"Dolna granica: {lower_bound:,.0f} PLN")
print(f"Górna granica: {upper_bound:,.0f} PLN")

# Znajdź outliers
outliers = df[(df["cena"] < lower_bound) | (df["cena"] > upper_bound)]
print(
    f"\n🎯 Znaleziono {len(outliers)} outlierów ({len(outliers)/len(df)*100:.1f}% danych)"
)

# Kilka przykładów outlierów
print("\nPrzykłady outlierów:")
print(outliers.head())


# Wizualizacja outlierów
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df["cena"])
plt.title("Box plot cen (z outlierami)")
plt.ylabel("Cena (PLN)")

plt.subplot(1, 2, 2)
plt.hist(df["cena"], bins=50, alpha=0.7, edgecolor="black")
plt.axvline(lower_bound, color="red", linestyle="--", label="Dolna granica")
plt.axvline(upper_bound, color="red", linestyle="--", label="Górna granica")
plt.title("Histogram cen z granicami outlierów")
plt.xlabel("Cena (PLN)")
plt.ylabel("Liczba domów")
plt.legend()

plt.tight_layout()
plt.show()
