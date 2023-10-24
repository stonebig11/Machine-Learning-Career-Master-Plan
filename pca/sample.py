# Import necessary libraries
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features

# Create a PCA model with 2 principal components (you can choose a different number)
pca = PCA(n_components=2)

# Fit and transform the data to the new feature space
X_pca = pca.fit_transform(X)

# Visualize the data in the new feature space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap=plt.cm.Paired)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA of Iris Dataset")
plt.show()
