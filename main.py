import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

dataset.head()

dataset['class'].unique()

X = zscore(dataset[['sepal-length', 'sepal-width']].values)
t = dataset['class'].values == 'Iris-setosa'
X, t = shuffle(X, t, random_state=0)

setosa = X[t]
not_setosa = X[~t]

plt.figure(figsize=(12, 7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.show()

perc_classifier = Perceptron(shuffle=False, random_state=0)
perc_classifier.fit(X, t)

plt.figure(figsize=(12, 7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

coef = perc_classifier.coef_.flatten()
w0 = perc_classifier.intercept_
w1 = coef[0]
w2 = coef[1]

step = 100
ds_x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), step)
ds_x2 = [-(w0 + w1 * x1) / w2 for x1 in ds_x1]
plt.plot(ds_x1, ds_x2, label='DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.show()

t_pred = perc_classifier.predict(X)

confusion_matrix(t, t_pred)

accuracy_score(t, t_pred)

precision_score(t, t_pred)

recall_score(t, t_pred)

f1_score(t, t_pred)

w = np.ones(3)
n_epochs = 10
for epochs in range(n_epochs):
    for i, (x_i, t_i) in enumerate(zip(X, t)):
        corr_t_i = 1 if t_i else -1
        ext_x = np.concatenate([np.ones(1), x_i.flatten()])
        if np.sign(w.dot(ext_x)) != corr_t_i:
            w = w + ext_x * corr_t_i

def plot_ds(X, w, step=100, label='DS'):

    ds_x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), step)
    ds_x2 = [-(w[0] + w[1] * x1) / w[2] for x1 in ds_x1]
    plt.plot(ds_x1, ds_x2, label=label)


plt.figure(figsize=(12, 7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

plot_ds(X, w, label='Implemented Perceptron DS')

coef = perc_classifier.coef_.flatten()
w0 = perc_classifier.intercept_
perc_w = np.array([w0, coef[0], coef[1]], dtype=object)
plot_ds(X, perc_w, label='Scikit-Learn Perceptron DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.show()

log_classifier = LogisticRegression(penalty='none')
log_classifier.fit(X, t)

plt.figure(figsize=(12, 7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', marker='x')

plot_ds(X, w, label='Implemented Perceptron DS')

coef = perc_classifier.coef_.flatten()
w0 = perc_classifier.intercept_
perc_w = np.array([w0, coef[0], coef[1]], dtype=object)
plot_ds(X, perc_w, label='Scikit-Learn Perceptron DS')

coef = log_classifier.coef_.flatten()
w0 = log_classifier.intercept_
log_w = np.array([w0, coef[0], coef[1]], dtype=object)
plot_ds(X, log_w, label='Scikit-Learn Logistic Regression DS')

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.show()


multi_t = dataset['class']
multi_log_classifier = LogisticRegression()
multi_log_classifier.fit(X, multi_t)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X, t)
t_pred = gnb_classifier.predict(X)

print(accuracy_score(t, t_pred))
print(recall_score(t, t_pred))
print(precision_score(t, t_pred))
print(confusion_matrix(t, t_pred))

# Creazione di nuovi dati

N = 100
new_samples = np.empty((N, 2))
new_t = np.empty(N, dtype=bool)

for i in range(N):
    class_ = np.random.choice([0, 1], p=gnb_classifier.class_prior_)
    new_t[i] = class_

    thetas = gnb_classifier.theta_[class_, :]

    sigmas = gnb_classifier.sigma_[class_, :]

    new_samples[i, 0] = np.random.normal(thetas[0], sigmas[0], 1)

    new_samples[i, 1] = np.random.normal(thetas[1], sigmas[1], 1)

new_setosa = new_samples[new_t, :]
new_not_setosa = new_samples[~new_t, :]

plt.figure(figsize=(12, 7))
plt.scatter(setosa[:, 0], setosa[:, 1], label='setosa', color='red')
plt.scatter(not_setosa[:, 0], not_setosa[:, 1], label='not setosa', color='blue')

plt.scatter(new_setosa[:, 0], new_setosa[:, 1], label='new, setosa', color='red', marker='x', alpha=0.3)
plt.scatter(new_not_setosa[:, 0], new_not_setosa[:, 1], label='new, not setosa', color='blue', marker='x', alpha=0.3)

plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
plt.legend()
plt.show()