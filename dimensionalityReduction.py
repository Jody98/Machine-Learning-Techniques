from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

iris = datasets.load_iris()

X = iris.data

X_tilde = X - np.mean(X, axis=0)
np.mean(X, axis=0)
S = np.dot(X_tilde.T, X_tilde)
print(S)
eigenvalues, eigeinvectors = np.linalg.eig(S)

print('eigenvalues:\n', eigenvalues)
print('eigenvectors:\n', eigeinvectors)

W = eigeinvectors
T = np.dot(X_tilde, W)
print('Original data point:', X[21])
print('Transformed data point:', T[21])

pca = PCA()
pca.fit(X)
print('First PC direction:', pca.components_[:, 0])
explained = pca.explained_variance_
print('Explained variance:', explained)

T = pca.transform(X)
print('The same data point as before:', T[21])

explained_variance = np.cumsum(explained)/sum(explained)
explained_variance = np.insert(explained_variance, 0, 0.)
plt.figure(figsize=(12, 7))
plt.plot(range(5), explained_variance)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.xticks(range(5))
plt.yticks(np.arange(0., 1.1, 0.1))
plt.grid()
plt.show()

explained_variance = np.cumsum(explained)/sum(explained)
X_tilde = X[:, explained_variance < 0.98]

pca2 = PCA(n_components=2)
T_12 = pca2.fit_transform(X)
print('Data point with reduced dimensionality', T_12[21])

