from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)


# domyślne wartośći dla brakujących danych w kolumnach 'garaz' i 'balkon' to False:
df["garaz"] = df["garaz"].fillna(False)
df["balkon"] = df["balkon"].fillna(False)


# -----------------------------------
# ZADANIE 4.1: Tworzenie nowych cech numerycznych
print("\n" + "=" * 50)
print("=== INŻYNIERIA CECH NUMERYCZNYCH ===")

#  Stwórz cechę 'cena_za_m2' (cena podzielona przez powierzchnię)
df["cena_za_m2"] = (df["cena"] / df["powierzchnia"]).round(2)

# Stwórz cechę 'metry_na_pokoj' (powierzchnia podzielona przez liczbę pokoi)
df["metry_na_pokoj"] = (df["powierzchnia"] / df["liczba_pokoi"]).round(2)

# Stwórz cechę 'wiek_kategoria' (nowy: <5 lat, średni: 5-20 lat, stary: >20 lat)
df["wiek_kategoria"] = pd.cut(
    df["wiek_domu"], bins=[-1, 5, 20, float("inf")], labels=["nowy", "średni", "stary"]
)

print("Nowe cechy numeryczne:")
print(f"Średnia cena za m²: {df['cena_za_m2'].mean():,.0f} PLN/m²")
print(f"Średnie metry na pokój: {df['metry_na_pokoj'].mean():.1f} m²/pokój")

print("\nRozkład kategorii wieku:")
print(df["wiek_kategoria"].value_counts().to_dict())


# -----------------------------------
# ZADANIE 4.2: Tworzenie cech binarnych
print("\n\n" + "=" * 50)
print("=== INŻYNIERIA CECH BINARNYCH ===")

# Stwórz cechę 'czy_nowy' (dom mladszy niż 5 lat)
df["czy_nowy"] = df["wiek_domu"] < 5

# Stwórz cechę 'blisko_centrum' (bliżej niż 10 km od centrum)
df["blisko_centrum"] = df["odleglosc_centrum"] < 10

# Stwórz cechę 'duzy_dom' (większy niż mediana powierzchni)
mediana_powierzchni = df["powierzchnia"].median()
df["duzy_dom"] = df["powierzchnia"] > mediana_powierzchni

# Stwórz cechę 'premium' (ma garaż, balkon i jest w Śródmieściu/Mokotowie)
df["premium"] = (
    df["garaz"] & df["balkon"] & df["dzielnica"].isin(["Śródmieście", "Mokotów"])
)


print("Rozkład nowych cech binarnych:")
binary_features = ["czy_nowy", "blisko_centrum", "duzy_dom", "premium"]
for feature in binary_features:
    print(f"{feature}: {df[feature].sum()} domów ({df[feature].mean()*100:.1f}%)")


# -----------------------------------
# ZADANIE 4.3: Encoding zmiennych kategorycznych
print("\n\n" + "=" * 50)
print("=== ENCODING KATEGORII ===")

# Stwórz dummy variables dla kolumny 'dzielnica'
dzielnica_dummies = pd.get_dummies(df["dzielnica"], prefix="dzielnica")

print("Dummy variables dla dzielnic:")
print(dzielnica_dummies.head())

# Dodaj dummy variables do głównego DataFrame
df = pd.concat([df, dzielnica_dummies], axis=1)

print(f"\n📏 Rozmiar DataFrame po dodaniu cech: {df.shape}")
print("\nNowe kolumny:")
print(df.columns.tolist())


print("\n\n" + "=" * 50)
print("=== OSTATECZNY KSZTAŁT DATA FRAME ===")
print(df.head())


# PODSUMOWANIE ŚLEDZTWA
print("\n\n" + "=" * 50)
print("🕵️ === RAPORT DETEKTYWA === 🕵️")
print(f"📊 Przeanalizowaliśmy {len(df)} domów w Warszawie")
print(f"🏠 Średnia cena: {df['cena'].mean():,.0f} PLN")
print(f"📐 Średnia powierzchnia: {df['powierzchnia'].mean():.0f} m²")
print(f"💰 Średnia cena za m²: {df['cena_za_m2'].mean():,.0f} PLN/m²")

print("\n🏆 TOP 3 NAJDROŻSZE DZIELNICE:")
cena_wg_dzielnicy = df.groupby("dzielnica")["cena"].mean().sort_values(ascending=False)
for i, (dzielnica, cena) in enumerate(cena_wg_dzielnicy.head(3).items(), 1):
    print(f"{i}. {dzielnica}: {cena:,.0f} PLN")

print("\n🔑 NAJWAŻNIEJSZE ODKRYCIA:")
correlation_with_price = (
    df.select_dtypes(include=[np.number])
    .corr()["cena"]
    .abs()
    .sort_values(ascending=False)
)
print("Czynniki najbardziej wpływające na cenę:")
for feature, corr in correlation_with_price.head(6).items():
    if feature != "cena":
        print(f"  • {feature}: {corr:.3f}")

print("\n✅ Dane są gotowe do budowy modelu ML!")


# BONUS
print("\n\n" + "=" * 50)
print("🚀 === CHALLENGE === 🚀")

# Stwórz wykres pokazujący rozkład cen za m² w każdej dzielnicy (użyj violin plot lub multiple histograms)
# plt.figure(figsize=(12, 6))
# sns.violinplot(data=df, x="dzielnica", y="cena_za_m2", inner="quartile")
# plt.title("Rozkład cen za m² w każdej dzielnicy")
# plt.xlabel("Dzielnica")
# plt.ylabel("Cena za m² [PLN]")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()


# Znajdź dom o najwyższej i najniższej cenie za m² i sprawdź dlaczego tak jest
dom_najwyzsza_cena_za_m2 = df.loc[df["cena_za_m2"].idxmax()]
dom_najnizsza_cena_za_m2 = df.loc[df["cena_za_m2"].idxmin()]

print("\nDom o najwyższej cenie za m²:")
print(dom_najwyzsza_cena_za_m2)

print("\nDom o najniższej cenie za m²:")
print(dom_najnizsza_cena_za_m2)

# Stwórz "indeks luksusu" dla każdego domu
# (kombinacja powierzchni, lokalizacji, wieku, garażu, balkonu)
# Indeks luksusu z znormalizowanymi cechami
df["indeks_luksusu"] = (
    df["duzy_dom"].astype(float) * 0.3
    + df["czy_nowy"].astype(float) * 0.3
    + df["garaz"].astype(float) * 0.1
    + df["balkon"].astype(float) * 0.1
    + df["blisko_centrum"].astype(float) * 0.2
).round(2)


# 💾 Zapisz wyczyszczone dane do pliku CSV
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa_clean.csv"
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Dane zapisane jako '{OUTPUT_PATH}'")
