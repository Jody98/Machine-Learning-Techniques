import numpy as np
import numpy.matlib

nS = 3
nSA = 5
P_sas = np.array([[0.9, 0.1, 0.0],
                  [0.3, 0.7, 0.0],
                  [0.4, 0.6, 0.0],
                  [0.0, 0.3, 0.7],
                  [0.2, 0.0, 0.8]])

R_sa = np.array([0.9*0.0 + 0.1*20,
                 0.3*(-2) + 0.7*(-27),
                 0.4*0.0 + 0.6*20,
                 0.3*(-5) + 0.7*(-100),
                 0.2*0.0 + 0.8*50])

mu = np.array([1.0, 0.0, 0.0])
gamma = 0.9

pi = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0]])

############################################################
#PREDIZIONE

V = np.linalg.inv(np.eye(nS) - gamma * pi @ P_sas) @ (pi @ R_sa)
print('State value function:\n', V)

eigenvalues, _ = np.linalg.eig(pi @ P_sas)
print('The eigenvalues of (pi * P_sas) are:\n', eigenvalues)
eigenvalues, _ = np.linalg.eig(gamma * pi @ P_sas)
print('The eigenvalues of (gamma * pi * P_sas) are:\n', eigenvalues)
eigenvalues, _ = np.linalg.eig(np.eye(nS) - gamma * pi @ P_sas)
print('The eigenvalues of (I - gamma * pi * P_sas) are:\n', eigenvalues)


V_old = np.zeros(nS)
tol = 0.0001
V = pi @ R_sa
while np.any(np.abs(V_old - V) > tol):
    V_old = V
    V = pi @ (R_sa + gamma * P_sas @ V)
print('State value function:\n', V)

Q = np.linalg.inv(np.eye(nSA) - gamma * P_sas @ pi) @ R_sa
print('State-Action Value function:\n', Q)

Q_old = np.zeros(nSA)
tol = 0.0001
Q = R_sa
while np.any(np.abs(Q_old - Q) > tol):
    Q_old = Q
    V = R_sa + gamma * P_sas @ pi @ Q
print('State-Action Value function:\n', Q)

pi_myo = np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]])

pi_far = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]])

gammas = [0.5, 0.9, 0.99]

for gamma in gammas:
    V_myo = np.linalg.inv(np.eye(nS) - gamma * pi_myo @ P_sas) @ (pi_myo @ R_sa)
    V_far = np.linalg.inv(np.eye(nS) - gamma * pi_far @ P_sas) @ (pi_myo @ R_sa)
    print('gamma:', gamma)
    print('V_myo:', V_myo)
    print('V_far:', V_far, '\n')

##############################################################
#CONTROLLO
policies = []
policies.append(np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0]]))
policies.append(np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0]]))
policies.append(np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0]]))
policies.append(np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0]]))

gamma = 0.9
V_max = np.zeros(nS)
i_max = -1
for i, pi in enumerate(policies):
    V = np.linalg.inv(np.eye(nS) - gamma * pi @ P_sas) @ (pi @ R_sa)
    print('Value of policy', i, 'is:', V)
    if np.all(V > V_max):
        V_max = V
        i_max = i

print('\nThe optimal policy is the', i_max, 'one:\n', policies[i_max])

adm_actions = np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])

pi = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0]])

Q = np.zeros(nSA)
Q_old = np.ones(nSA)
while np.any(Q != Q_old):
    Q_old = Q
    Q = np.linalg.inv(np.eye(nSA) - gamma * P_sas @ pi) @ R_sa
    greedy_rep = np.matlib.repmat(Q, nS, 1) * adm_actions
    greedy_rep[greedy_rep == 0] = -np.inf
    greedy_actions = [[i, np.argmax(greedy_rep[i, :])] for i in range(nS)]
    pi = np.array([1. if [x, y] in greedy_actions else 0.
                   for x, y in np.ndindex((nS, nSA))]).reshape(nS, nSA)
print('The optimal policy is:\n', pi)

V = np.zeros(nS)
V_old = np.ones(nS)
tol = 0.0001
while np.any(np.abs(V_old - V) > tol):
    V_old = V
    greedy_rep = R_sa + gamma * P_sas @ V
    greedy_rep = np.matlib.repmat(greedy_rep, nS, 1) * adm_actions
    greedy_rep[greedy_rep == 0] = -np.inf
    V = np.amax(greedy_rep, axis=1)

print('The optimal value function is:\n', V)

greedy_rep = R_sa + gamma * P_sas @ V
greedy_rep = np.matlib.repmat(greedy_rep, nS, 1) * adm_actions
greedy_rep[greedy_rep == 0] = -np.inf
greedy_actions = [[i, np.argmax(greedy_rep[i, :])] for i in range(nS)]
pi = np.array([1. if [x, y] in greedy_actions else 0.
               for x, y in np.ndindex((nS, nSA))]).reshape(nS, nSA)
print('The optimal policy is:\n', pi)

