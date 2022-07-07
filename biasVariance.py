import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

n_points = 1000
eps = 0.7


def fun(x):
    return 1 + 1/2 * x + 1/10 * x**2


x = np.random.uniform(low=0, high=5, size=(n_points, 1))

t = fun(x)
t_noisy = t + eps * np.random.randn(n_points, 1)
phi = np.concatenate([x, x**2], axis=1)

lin_model = linear_model.LinearRegression()
lin_model.fit(x, t_noisy)

qua_model = linear_model.LinearRegression()
qua_model.fit(phi, t_noisy)

plt.figure(figsize=(16, 9))
plt.scatter(x, t_noisy, label='true')

plt.plot(x, lin_model.predict(x), label='linear model', color='red')
plt.scatter(x, qua_model.predict(phi), label='quadratic model', color='orange')

plt.show()

real_par = np.array([1, 1/2, 1/10])
best_lin_par = np.array([7/12, 1, 0])

lin_c = np.array([lin_model.intercept_[0], lin_model.coef_[0][0], 0])
qua_c = np.concatenate([qua_model.intercept_, qua_model.coef_.flatten()])

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection='3d')
s = 70

ax.scatter(*real_par, marker='x', color='blue', s=s, label='Best Quadratic')
ax.scatter(*best_lin_par, marker='o', color='red', s=s, label='Best Linear')
ax.scatter(*qua_c, marker='+', color='blue', s=s, label='Fitted Quadratic')
ax.scatter(*lin_c, marker='+', color='red', s=s, label='Fitted Linear')

ax.set_xlabel('w_0')
ax.set_ylabel('w_1')
ax.set_zlabel('w_2')
ax.view_init(elev=20, azim=32)
plt.title('Parameter Space')
plt.legend()
plt.grid()
plt.show()


def sample_and_fit(n_points):
    x2 = np.random.uniform(low=0, high=5, size=(n_points, 1))
    t2 = fun(x2)
    t_noisy2 = t2 + eps * np.random.randn(n_points, 1)
    phi2 = np.concatenate([x2, x2 ** 2], axis=1)
    lin_model2 = linear_model.LinearRegression()
    lin_model2.fit(x2, t_noisy2)
    qua_model2 = linear_model.LinearRegression()
    qua_model2.fit(phi2, t_noisy2)
    return lin_model2, qua_model2


def multiple_sample_and_fit(n_repetitions, n_points):
    lin_coeff = np.zeros((n_repetitions, 3))
    qua_coeff = np.zeros((n_repetitions, 3))

    for i in range(n_repetitions):
        lin_model2, qua_model2 = sample_and_fit(n_points)
        lin_coeff[i, :] = np.concatenate([lin_model2.intercept_, lin_model2.coef_.flatten(), np.zeros(1)])
        qua_coeff[i, :] = np.concatenate([qua_model2.intercept_, qua_model2.coef_.flatten()])
    return lin_coeff, qua_coeff


def plot_models(lin_coeff, qua_coeff):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(qua_coeff[:, 0], qua_coeff[:, 1], qua_coeff[:, 2], marker='.', color='blue', s=s, alpha=0.3)
    ax.scatter(lin_coeff[:, 0], lin_coeff[:, 1], lin_coeff[:, 2], marker='.', color='red', s=s, alpha=0.3)

    ax.scatter(*real_par, marker='x', color='blue', s=100)
    ax.scatter(*best_lin_par, marker='o', color='red', s=100)
    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    ax.set_zlabel('w_2')
    ax.view_init(elev=20, azim=32)
    plt.title('Parameter Space')
    plt.grid()
    plt.show()


n_repetitions = 100
n_points = 1000
lin_coeff, qua_coeff = multiple_sample_and_fit(n_repetitions, n_points)
plot_models(lin_coeff, qua_coeff)

n_points = 1
x_new = np.random.uniform(low=0, high=5, size=(n_points, 1))

t_new = fun(x_new)
t_new_noisy = t_new + eps * np.random.randn(n_points, 1)

x_enh_new = np.array([1, x_new, 0])
phi_enh_new = np.array([1, x_new, x_new**2])

y_pred_lin = lin_coeff @ x_enh_new
y_pred_qua = qua_coeff @ phi_enh_new

error_lin = np.mean((t_new_noisy - y_pred_lin)**2)
bias_lin = np.mean(t_new - y_pred_lin)**2
variance_lin = np.var(y_pred_lin)
var_t_lin = error_lin - variance_lin - bias_lin

error_qua = np.mean((t_new_noisy - y_pred_qua)**2)
bias_qua = np.mean(t_new - y_pred_qua)**2
variance_qua = np.var(y_pred_qua)
var_t_qua = error_qua - variance_qua - bias_qua

print("--Single Point--")
print('Linear Error: {}'.format(error_lin[0][0]))
print('Linear Bias: {}'.format(bias_lin[0][0]))
print('Linear Variance: {}'.format(variance_lin[0][0]))
print('Linear Sigma: {}'.format(var_t_lin[0][0]))

print("--Single Point--")
print('Quadratic Error: {}'.format(error_qua[0][0]))
print('Quadratic Bias: {}'.format(bias_qua[0][0]))
print('Quadratic Variance: {}'.format(variance_qua[0][0]))
print('Quadratic Sigma: {}'.format(var_t_qua[0][0]))
