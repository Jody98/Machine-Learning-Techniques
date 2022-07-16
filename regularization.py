from sklearn import datasets
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(iris.feature_names)

np.random.seed(0)
y[y == 2] = 0
D = np.concatenate((X[:, :2], y.reshape(len(y), -1)), axis=1)
np.random.shuffle(D)
X_train = D[:100, :2]
y_train = D[:100, 2]
X_test = D[100:150, :2]
y_test = D[100:150, 2]

versicolor = X_train[y_train == 1]
not_versicolor = X_train[y_train == 0]

plt.figure(figsize=(12, 7))
plt.scatter(versicolor[:, 0], versicolor[:, 1], label='versicolor')
plt.scatter(not_versicolor[:, 0], not_versicolor[:, 1], label='not versicolor', marker='x')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.grid()
plt.legend()
plt.show()

def knn_decision_surface(X, y, clf, k):
    h = .02
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 0].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['orange', 'cornflowerblue'])

    plt.figure(figsize=(12,7))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)
    plt.scatter(versicolor[:, 0], versicolor[:, 1], label='versicolor')
    plt.scatter(not_versicolor[:, 0], not_versicolor[:, 1], label='not_versicolor', marker='x')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title("k = {}".format(k))
    plt.show()

for k in np.arange(1, 11):
    knn_classifier = neighbors.KNeighborsClassifier(k)
    knn_classifier.fit(X_train[:, :2], y_train)
    knn_decision_surface(X, y, knn_classifier, k)


parameters = np.arange(1, 11)
train_accuracy = []
test_accuracy = []

for k in parameters:
    knn_classifier = neighbors.KNeighborsClassifier(k)
    knn_classifier.fit(X_train[:, :2], y_train)

    y_hat = knn_classifier.predict(X_train)
    accuracy = sum(y_hat == y_test)/len(y_test)
    train_accuracy.append(accuracy)

    y_hat = knn_classifier.predict(X_test)
    accuracy = sum(y_hat == y_test) / len(y_test)
    test_accuracy.append(accuracy)


plt.figure(figsize=(12, 7))
plt.plot(parameters, train_accuracy, label='train accuracy')
plt.plot(parameters, test_accuracy, label='test accuracy')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend()
plt.show()