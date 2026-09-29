from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "domy_warszawa.csv"
df = pd.read_csv(DATA_PATH)

# ZADANIE 3.2: Strategie radzenia sobie z brakami danych
print("=== NAPRAWIAMY BRAKI DANYCH ===")

# Tworzymy kopię danych do eksperymentów
df_clean = df.copy()

print("Braki PRZED czyszczeniem:")
print(df_clean.isnull().sum())

# Strategia 1 - wypełnij braki w kolumnie 'garaz' wartością False
# (zakładamy, że brak informacji = brak garażu)
df_clean["garaz"] = df_clean["garaz"].fillna(False)


# Strategia 2 - wypełnij braki w kolumnie 'balkon' na podstawie analizy
# Sprawdźmy czy domy nowsze częściej mają balkony
print("\n📊 Analiza: czy nowsze domy częściej mają balkony?")
print(df_clean.groupby("balkon")["wiek_domu"].mean())

# Na podstawie analizy, wypełnij braki
# Jeśli dom ma mniej niż 10 lat, prawdopodobnie ma balkon
mask = (df_clean["wiek_domu"] < 10) & (df_clean["balkon"].isna())
df_clean.loc[mask, "balkon"] = True


print("\nBraki PO czyszczeniu:")
print(df_clean.isnull().sum())

print("\n✅ Dane zostały wyczyszczone!")
