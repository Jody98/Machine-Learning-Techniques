from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

print('Originall targets\n', y)
y = iris.target.copy()
y[y == 1] = 0
y[y == 2] = 1
print('Iris-Virginica targets\n', y)

np.random.seed(0)
D = np.concatenate((X, y.reshape(len(y), -1)), axis=1)
np.random.shuffle(D)
X_train = D[:90, :4]
y_train = D[:90, 4]
X_vali = D[90:120, :4]
y_vali = D[90:120, 4]
X_test = D[120:150, :4]
y_test = D[120:150, 4]

log_classifier = LogisticRegression(penalty='none')
log_classifier.fit(X_train, y_train)
y_hat_vali = log_classifier.predict(X_vali)
vali_accuracy = sum(y_hat_vali == y_vali)/len(y_vali)
print('Validation Accuracy:', vali_accuracy)

F = [0, 1, 2, 3]
for i in range(len(F)):
    X_train_i = np.delete(X_train, i, axis=1)
    X_vali_i = np.delete(X_vali, i, axis=1)
    log_classifier_i = LogisticRegression(penalty='none')
    log_classifier_i.fit(X_train_i, y_train)
    y_hat_vali = log_classifier_i.predict(X_vali_i)
    vali_accuracy = sum(y_hat_vali == y_vali) / len(y_vali)
    print('The model with features:', F, 'without', F[i], 'has validation accuracy:', vali_accuracy)

X_train = np.concatenate((X_train, X_vali))
y_train = np.concatenate((y_train, y_vali))
F = [2, 3]
X_train_fs = X_train[:, F]
X_test_fs = X_test[:, F]
log_classifier_fs = LogisticRegression(penalty='none')
log_classifier_fs.fit(X_train_fs, y_train)
y_hat_test = log_classifier_fs.predict(X_test_fs)
test_accuracy = sum(y_hat_test == y_test)/len(y_test)
print('The model with features:', F, 'has test accuracy:', test_accuracy)

virginica = X[y == 1]
not_virginica = X[y == 0]
plt.figure(figsize=(12, 7))
plt.scatter(virginica[:, 2], virginica[:, 3], label='virginica')
plt.scatter(not_virginica[:, 2], not_virginica[:, 3], label='not virginica', marker='x')
plt.xlabel('x2')
plt.ylabel('x3')
plt.grid()
plt.legend()
plt.show()

def plot_ds(X, w, step=100, label='DS'):
    ds_x1 = np.linspace(X[:, 2].min(), X[:, 2].max(), step)
    ds_x2 = [-(w[0] + w[1]*x1)/w[2] for x1 in ds_x1]
    plt.plot(ds_x1, ds_x2, label=label)


plt.figure(figsize=(12, 7))
coef = log_classifier_fs.coef_.flatten()
w0 = log_classifier_fs.intercept_
log_w = np.array([w0, coef[0], coef[1]])
plot_ds(X, log_w, label='Logistic Regression')
plt.scatter(virginica[:, 2], virginica[:, 3], label='virginica')
plt.scatter(not_virginica[:, 2], not_virginica[:, 3], label='not virginica', marker='x')
plt.xlabel('x2')
plt.ylabel('x3')
plt.grid()
plt.legend()
plt.show()

