import numpy as np
import pandas as pd
from numpy.linalg import inv
from pandas.plotting import scatter_matrix
from scipy.stats import zscore
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from statsmodels import api as sm
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

dataset.head()

dataset = dataset.drop('class', axis=1)
dataset.head()

dataset.describe()

dataset.hist(figsize=(16, 9))
plt.show()

scatter_matrix(dataset, figsize=(16, 9))
plt.show()

dataset.plot.scatter('petal-length', 'petal-width', grid=True, figsize=(16, 9))

np.any(np.isnan(dataset.values))

x = zscore(dataset['petal-length'].values).reshape(-1, 1)
y = zscore(dataset['petal-width'].values)

lin_model = linear_model.LinearRegression()
lin_model.fit(x, y)

with plt.style.context('seaborn'):
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, label='true')

    w1 = lin_model.coef_
    w0 = lin_model.intercept_

    y_pred = lin_model.predict(x)

    plt.plot(x, y_pred, label='predicted', color='red')

    plt.legend(prop={'size':20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

# lin_model._residues()

r2_score(y, y_pred)

mean_squared_error(y, y_pred)

f_regression(x, y)

lin_model2 = sm.OLS(y, x).fit()
print(lin_model2.summary())

# lin_model2._results.params

# lin_model2._results.k_constant

n_samples = len(x)
Phi = np.ones((n_samples, 2))
Phi[:, 1] = x.flatten()
w = inv(Phi.T @ Phi) @ (Phi.T.dot(y))

ridge_model = linear_model.Ridge(alpha=10)
ridge_model.fit(x, y)

lasso_model = linear_model.Lasso(alpha=10)
lasso_model.fit(x, y)

with plt.style.context('seaborn'):
    plt.figure(figsize=(16, 9))
    plt.scatter(x, y, label='original samples')
    y_linear = [lin_model.coef_ * x_i + lin_model.intercept_ for x_i in x]
    plt.plot(x, y_linear, label='linear regression', color='red')
    for alpha in [0.1, 0.2, 0.5]:
        lasso_model = linear_model.Lasso(alpha=alpha)
        lasso_model.fit(x, y)
        y_lasso = [lasso_model.coef_ * x_i + lasso_model.intercept_ for x_i in x]
        plt.plot(x, y_lasso, label='lasso, alpha={}'.format(alpha))

    plt.legend(prop={'size':20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.show()

mean_squared_error(y, lasso_model.predict(x))
