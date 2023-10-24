# Import necessary libraries
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features

# Create a K-Means model with 3 clusters (you can choose a different number)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the K-Means model to the data
kmeans.fit(X)

# Get cluster assignments for each data point
labels = kmeans.labels_

# Visualize the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering of Iris Dataset")
plt.legend()
plt.show()
