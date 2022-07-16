import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

exp_reward = np.array([0.15, 0.1, 0.1, 0.2, 0.35, 0.2])
n_arms = len(exp_reward)
opt = np.max(exp_reward)
idx_opt = np.argmax(exp_reward)
deltas = opt - exp_reward
deltas = np.array([delta for delta in deltas if delta > 0])

T = 5000

def UCB1():
    ucb1_criterion = np.zeros(n_arms)
    expected_payoffs = np.zeros((n_arms))
    number_of_pulls = np.zeros(n_arms)

    regret = np.array([])
    pseudo_regret = np.array([])

    for t in range (1, T + 1):
        if t < n_arms:
            pulled_arm = t
        else:
            idxs = np.argwhere(ucb1_criterion == ucb1_criterion.max()).reshape(-1)
            pulled_arm = np.random.choice(idxs)

        reward = np.random.binomial(1, exp_reward[pulled_arm])
        if pulled_arm != idx_opt:
            reward_opt = np.random.binomial(1, exp_reward[idx_opt])
        else:
            reward_opt = reward

        number_of_pulls[pulled_arm] = number_of_pulls[pulled_arm] + 1
        expected_payoffs[pulled_arm] = ((expected_payoffs[pulled_arm] * (number_of_pulls[pulled_arm] - 1.0) + reward) /
                                        number_of_pulls[pulled_arm])
        for k in range(0, n_arms):
            ucb1_criterion[k] = expected_payoffs[k] + np.sqrt(2 * np.log(t)/number_of_pulls[k])

        regret = np.append(regret, reward_opt - reward)
        pseudo_regret = np.append(pseudo_regret, opt - exp_reward[pulled_arm])
    return regret, pseudo_regret

regret, pseudo_regret = UCB1()

plt.figure(figsize=(16,9))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(pseudo_regret), color='r', label='Pseudo-regret')
plt.plot(np.cumsum(regret), color='g', label='Regret')
plt.legend()
plt.grid()
plt.show()


n_repetitions = 5
regrets, pseudo_regrets = np.zeros((n_repetitions, T)), np.zeros((n_repetitions, T))
for i in range(n_repetitions):
    regrets[i], pseudo_regrets[i] = UCB1()

cumu_regret = np.cumsum(regrets, axis=1)
cumu_pseudo_regret = np.cumsum(pseudo_regrets, axis=1)
avg_cumu_regret = np.mean(cumu_regret, axis=0)
avg_cumu_pseudo_regret = np.mean(cumu_pseudo_regret, axis=0)
std_cumu_regret = np.std(cumu_regret, axis=0)
std_cumu_pseudo_regret = np.std(cumu_pseudo_regret, axis=0)

ucb1_upper_bound = np.array([8*np.log(t)*sum(1/deltas) + (1 + np.pi**2/3)*sum(deltas)
                             for t in range(1,T+1)])

plt.figure(figsize=(16,9))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(avg_cumu_pseudo_regret, color='r', label='Pseudo-regret')

plt.plot(avg_cumu_regret + 1.96 * std_cumu_regret / np.sqrt(n_repetitions), linestyle='--', color='g')
plt.plot(avg_cumu_regret, color='g', label='Regret')
plt.plot(avg_cumu_regret - 1.96 * std_cumu_regret / np.sqrt(n_repetitions), linestyle='--', color='g')
plt.plot(ucb1_upper_bound, color='b', label='Upper bound')

plt.legend()
plt.grid()
plt.show()


def thompson_sampling():
    T = 5000
    beta_parameters = np.ones((n_arms, 2))
    regret = np.array([])
    pseudo_regret = np.array([])
    for t in range(1, T + 1):
        samples = np.random.beta(beta_parameters[:, 0], beta_parameters[:, 1])
        pulled_arm = np.argmax(samples)
        reward = np.random.binomial(1, exp_reward[pulled_arm])
        if pulled_arm != idx_opt:
            reward_opt = np.random.binomial((1, exp_reward[idx_opt]))
        else:
            reward_opt = reward

        beta_parameters[pulled_arm, 0] = beta_parameters[pulled_arm, 0] + reward
        beta_parameters[pulled_arm, 1] = beta_parameters[pulled_arm, 1] + 1.0 - reward

        regret = np.append(regret, reward_opt - reward)
        pseudo_regret = np.append(pseudo_regret, opt - exp_reward[pulled_arm])
    return regret, pseudo_regret, beta_parameters

regret, pseudo_regret, beta_parameters = thompson_sampling()
print(beta_parameters)

plt.figure(figsize=(16, 9))
x = np.linspace(0,1,1000)
colors = ['red', 'green', 'blue', 'purple', 'orange']
for params, rew, color in zip(beta_parameters, exp_reward, colors):
  rv = beta(*params)
  plt.plot(x, rv.pdf(x), color=color)
  plt.axvline(rew, linestyle='--', color=color)
plt.grid()
plt.show()

plt.figure(figsize=(16,9))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(pseudo_regret), label='Pseudo-regret', color='r')
plt.plot(np.cumsum(regret), label='Regret', color='g')
plt.legend()
plt.grid()
plt.show()

n_repetitions = 5
regrets, pseudo_regrets = np.zeros((n_repetitions, T)), np.zeros((n_repetitions, T))
for i in range(n_repetitions):
  regrets[i], pseudo_regrets[i], _ = thompson_sampling()

# Compute the cumulative sum
cumu_regret = np.cumsum(regrets, axis=1)
cumu_pseudo_regret = np.cumsum(pseudo_regrets, axis=1)

# Take the average over different runs
avg_cumu_regret = np.mean(cumu_regret, axis=0)
avg_cumu_pseudo_regret = np.mean(cumu_pseudo_regret, axis=0)

std_cumu_regret = np.std(cumu_regret, axis=0)
std_cumu_pseudo_regret = np.std(cumu_pseudo_regret, axis=0)

plt.figure(figsize=(16,9))
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(avg_cumu_pseudo_regret, color='r', label='pseudo-regret')
plt.plot(avg_cumu_regret + std_cumu_regret, linestyle='--', color='g')
plt.plot(avg_cumu_regret, color='g', label='regret')
plt.plot(avg_cumu_regret - std_cumu_regret, linestyle='--', color='g')
plt.legend()
plt.grid()
plt.show()


