import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data.simple import labels, texts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from utils import clean_text

texts_cleaned = [clean_text(t) for t in texts]

# wektoryzacja
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts_cleaned)

# przygotowanie etykiet
y = np.array([label if label is not None else -1 for label in labels], dtype=object)

# trenowanie modelu SelfTrainingClassifier
# 💡 Ćwiczenie: Zmień bazowy model np. na KNeighborsClassifier() i zobacz, czy lepiej działa!
base_model = LogisticRegression()
self_training_model = SelfTrainingClassifier(base_model)
self_training_model.fit(X, y)

y_pred = self_training_model.predict(X)

# wizualizacja wyników
print("\n=== Raport klasyfikacji ===")
print(
    classification_report(
        [l if l is not None else "Nieznane" for l in labels], y_pred, zero_division=0
    )
)

cm = confusion_matrix(
    [l if l is not None else "Nieznane" for l in labels],
    y_pred,
    labels=["Sprzęt", "Konto", "Oprogramowanie", "Nieznane"],
)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Sprzęt", "Konto", "Oprogramowanie", "Nieznane"],
    yticklabels=["Sprzęt", "Konto", "Oprogramowanie", "Nieznane"],
)
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.title("Macierz pomyłek")
plt.show()
