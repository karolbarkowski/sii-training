import matplotlib.pyplot as plt
import seaborn as sns
from data.simple import tickets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils import clean_input, split_data

input_raw = [ticket["text"] for ticket in tickets]
labels = [ticket["label"] for ticket in tickets]

# czyszczenie i lematyzacja
input_cleaned = clean_input(input_raw)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = split_data(input_cleaned, labels, test_size=0.3)

# Przekształcenie tekstu na liczby (TF-IDF)
# 💡 Ćwiczenie: Użyj parametru stop_words='english' i zobacz, czy coś się zmieni (nawet jeśli dane są po polsku).
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Trenowanie modelu klasyfikacyjnego
# 💡 Ćwiczenie: Zamień LogisticRegression() na DecisionTreeClassifier() i porównaj wyniki.
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predykcja
y_pred = model.predict(
    X_test_vec,
)

# Ewaluacja
print("\n=== Raport klasyfikacji ===")
print(classification_report(y_test, y_pred))

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
