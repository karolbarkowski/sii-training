from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)

# -----------------------------------
# Zadanie 1.1: Podstawowe informacje o datasecie
print("\n" + "=" * 50)
print("Pierwsze 5 wierszy:")
print(df.head(5))

print("\n" + "=" * 50)
print("Kształt danych (wiersze, kolumny):")
print(df.shape)

print("\n" + "=" * 50)
print("Informacje o kolumnach:")
print(df.info())

# -----------------------------------
# Zadanie 1.2: Statystyki opisowe
print("\n" + "=" * 50)
print("Statystyki opisowe:")
print(df.describe())

print("\n" + "=" * 50)
print("Rozkład domów według dzielnic:")
print(df["dzielnica"].value_counts())

print("\n" + "=" * 50)
print("Rozkład domów z garażem:")
print(df["garaz"].value_counts())

print("\n" + "=" * 50)
print("Rozkład domów z balkonem:")
print(df["balkon"].value_counts())

print("\n" + "=" * 50)
print("Rozkład domów z garażem i balkonem:")
print(df[["garaz", "balkon"]].value_counts())

# -----------------------------------
# Zadanie 1.3: Identyfikacja braków danych
print("\n" + "=" * 50)
print("Liczba braków w każdej kolumnie:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("Procent braków w każdej kolumnie:")
print(((df.isnull().sum() / len(df)) * 100).round(2))
