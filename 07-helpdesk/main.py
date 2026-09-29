from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

DATA_PATH = Path(__file__).resolve().parent / "data" / "zgloszenia_helpdesk.csv"
df = pd.read_csv(DATA_PATH)


print(f"📊 {len(df)} zgłoszeń, {df.tekst.nunique()} unikalnych treści")
print(f"\nRozkład kategorii:")
print(df.kategoria.value_counts().to_string())


X_tr, X_te, y_tr, y_te = train_test_split(
    df.tekst, df.kategoria, test_size=0.25, random_state=42, stratify=df.kategoria
)
print(f"\n✂️  Train: {len(X_tr)} | Test: {len(X_te)}")


# Rok 2015: Bag of Words
vectorizer = TfidfVectorizer()
X_tr_bow = vectorizer.fit_transform(X_tr)
X_te_bow = vectorizer.transform(X_te)

model_bow = LogisticRegression(max_iter=1000).fit(X_tr_bow, y_tr)
acc_bow = accuracy_score(y_te, model_bow.predict(X_te_bow))

print(f"📖 Słownik: {len(vectorizer.vocabulary_)} unikalnych słów")
print(f"📐 Każde zgłoszenie to wektor o {X_tr_bow.shape[1]} wymiarach")
print(f"\n🎯 Accuracy (wszystkie przykłady treningowe): {acc_bow:.3f}")


# Rok 2020: embeddingi
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
encoder = AutoModel.from_pretrained(MODEL)


def embeduj(teksty, batch=64):
    """Zamienia listę zdań na wektory znaczeniowe (znormalizowane)."""
    out = []
    teksty = list(teksty)
    for i in range(0, len(teksty), batch):
        b = tokenizer(
            teksty[i : i + batch],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.no_grad():
            o = encoder(**b)
        maska = b["attention_mask"].unsqueeze(-1).float()
        e = (o.last_hidden_state * maska).sum(1) / maska.sum(
            1
        )  # uśrednienie po słowach
        out.append(e.numpy())
    E = np.vstack(out)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


print("⏳ Liczę embeddingi...")
E_tr, E_te = embeduj(X_tr), embeduj(X_te)
print(f"✅ Każde zgłoszenie to teraz wektor o {E_tr.shape[1]} wymiarach")
print(f"   (Bag of Words potrzebował {X_tr_bow.shape[1]} — i większość to zera)")
