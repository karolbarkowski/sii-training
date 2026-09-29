# Wczytanie większego zbioru
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils import apply_lemma, clean_text

DATA_PATH = Path(__file__).resolve().parent / "data" / "large_tickets.csv"
df = pd.read_csv(DATA_PATH)

# Czyszczenie i lematyzacja
df["text_clean"] = df["text"].apply(clean_text).apply(apply_lemma)

# Podział na trening/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"],
)

# Wektoryzacja
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Trenowanie
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Ewaluacja
y_pred = model.predict(X_test_vec)
print("\n=== Raport klasyfikacji na dużym zbiorze ===")
print(classification_report(y_test, y_pred, zero_division=0))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_,
)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.title("Macierz pomyłek")
plt.show()
