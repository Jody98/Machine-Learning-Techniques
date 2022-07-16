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

env = Environment()
T = 10
a = 0
ret = 0
s = env.reset()
print('step', 0, 's:', s)
for i in range(T):
    s, r = env.transition_model(a)
    ret = ret + r * (env.gamma ** (i + 1))
    print('\nstep:', i + 1, 's:', s, 'a:', a, 'r:', r)
print('\nthe final return is:', ret)


pi_far = np.array([[0., 1., 0.],
                   [0., 0., 1.],
                   [1., 0., 0.]])

env = Environment()
M = 5000
m = 1
V = np.zeros(env.nS)
actions = [0, 1, 2]
s = env.reset()

while m < M:
    alpha = (1 - m/M)
    a = np.random.choice(actions, p=pi_far[s])
    s_prime, r = env.transition_model(a)
    V[s] = V[s] + alpha * (r + env.gamma * V[s_prime] - V[s])
    m = m + 1
    s = s_prime
print('The final V function is:\n', V)


