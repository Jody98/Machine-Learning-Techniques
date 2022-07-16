import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

dataset = datasets.load_iris()
x = zscore(dataset.data[:, 2]).reshape(-1, 1)
y = zscore(dataset.data[:, 3])

phi = 3
l = 0.02
sigma_sq = 0.2
kernel = ConstantKernel(phi, constant_value_bounds="fixed") * RBF(l, length_scale_bounds="fixed")
gpr = GaussianProcessRegressor(kernel=kernel, alpha=sigma_sq).fit(x, y)
x_pred = np.array(x)
x_pred = np.sort(x_pred, axis=0)
y_pred, sigma = gpr.predict(x_pred, return_std=True)
plt.figure(figsize=(12, 7))
plt.plot(x, y, 'r.', markersize=10, label='Observations')
plt.plot(x_pred, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='upper left')
plt.show()
