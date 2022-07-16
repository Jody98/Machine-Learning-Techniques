import numpy as np

class Environment(object):

    def __init__(self):
        # states and actions
        self.nS = 3
        self.nA = 3
        self.allowed_actions = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0]])
        # initial state distribution and discount factor
        self.mu = np.array([1., 0., 0.])
        self.gamma = 0.9
        # transition model (SA rows, S columns)
        self.P = np.array([[0.9, 0.1, 0.],
                          [0.3, 0.7, 0.],
                          [0., 0., 0.],
                          [0.4, 0.6, 0.],
                          [0., 0., 0.],
                          [0., 0.3, 0.7],
                          [0.2, 0, 0.8],
                          [0., 0., 0.],
                          [0., 0., 0.]])
        # immediate reward (SA rows, S columns)
        self.R = np.array([[0, 20, 0],
                          [-2, -27, 0],
                          [0., 0., 0.],
                          [0, 20, 0],
                          [0., 0., 0.],
                          [0, -5, -100],
                          [0, 0, 50],
                          [0., 0., 0.],
                          [0., 0., 0.]])

    def _seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.s = s = np.random.choice(self.nS, p=self.mu)
        return s

    def transition_model(self, a):
        sa = self.s * self.nA + a
        self.s = s_prime = np.random.choice(self.nS, p=self.P[sa, :])
        inst_rew = self.R[sa, s_prime]
        return s_prime, inst_rew

def eps_greedy(s, Q, eps, allowed_actions):
    if np.random.rand() <= eps:
        actions = np.where(allowed_actions)
        actions = actions[0]
        a = np.random.choice(actions, p=(np.ones(len(actions))/len(actions)))
    else:
        Q_s = Q[s, :].copy()
        Q_s[allowed_actions == 0] = -np.inf
        a = np.argmax(Q_s)
    return a

env = Environment()
env._seed(10)
M = 5000
m = 1
Q = np.zeros((env.nS, env.nA))
s = env.reset()
a = eps_greedy(s, Q, 1.0, env.allowed_actions[s])
while m < M:
    alpha = (1 - m/M)
    eps = (1 - m/M) ** 2
    s_prime, r = env.transition_model(a)
    a_prime = eps_greedy(s, Q, eps, env.allowed_actions[s_prime])
    Q[s, a] = Q[s, a] + alpha * (r + env.gamma * Q[s_prime, a_prime] - Q[s, a])
    m = m + 1
    s = s_prime
    a = a_prime
print('The final Q function is:\n', Q)


env = Environment()
env._seed(10)
M = 5000
m = 1
Q = np.zeros((env.nS, env.nA))
s = env.reset()
a = eps_greedy(s, Q, 1.0, env.allowed_actions[s])
while m < M:
    alpha = (1 - m/M)
    eps = (1 - m/M) ** 2
    s_prime, r = env.transition_model(a)
    a_prime = eps_greedy(s, Q, eps, env.allowed_actions[s_prime])
    Q[s, a] = Q[s, a] + alpha * (r + env.gamma * np.max(Q[s_prime, :]) - Q[s, a])
    m = m + 1
    s = s_prime
    a = a_prime
print('The final Q function is:\n', Q)

