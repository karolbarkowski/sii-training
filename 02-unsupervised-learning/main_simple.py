import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data.simple import texts
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import clean_text

# czyszczenie tekstu
texts_cleaned = [clean_text(t) for t in texts]

# wektoryzacja tekstu (TF-IDF)
vectorizer = TfidfVectorizer(max_features=20)
X_vec = vectorizer.fit_transform(texts_cleaned)

# klasteryzacja KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_vec)
clusters = kmeans.labels_


# wizualizacja wyników klasteryzacji
df = pd.DataFrame({"Zgłoszenie": texts, "Grupa": clusters})
df = df.sort_values(by="Grupa")
print(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_vec.toarray())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis", s=100)
plt.title("Wizualizacja klastrów zgłoszeń")
plt.xlabel("Wymiar 1")
plt.ylabel("Wymiar 2")
plt.show()
