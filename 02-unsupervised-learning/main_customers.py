from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "fake_customers.csv"
df = pd.read_csv(DATA_PATH)

# kolumny, które będą używane do klasteryzacji
feature_columns = [
    "Age",
    "AnnualIncome",
    "AnnualSpend",
    "PurchasesLastYear",
    "AverageOrderValue",
    "MonthlyWebsiteVisits",
    "DaysSinceLastPurchase",
    "DiscountUsageRate",
    "ReturnsLastYear",
    "SupportTicketsLastYear",
    "YearsAsCustomer",
]

X = df[feature_columns]

# Standaryzacja cech przed klasteryzacją, aby kolumny o dużych wartościach nie dominowały.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

cluster_counts = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for axis, cluster_count in zip(axes, cluster_counts):
    # klasteryzacja
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # wizualizcja
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=clusters,
        palette="viridis",
        s=100,
        ax=axis,
        legend=False,
    )
    axis.set_title(f"KMeans z {cluster_count} klastrami")
    axis.set_xlabel("Wymiar 1")
    axis.set_ylabel("Wymiar 2")

plt.tight_layout()
plt.show()
