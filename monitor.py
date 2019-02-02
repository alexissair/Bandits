from epsgreedy import *
from ucb import *
import matplotlib.pyplot as plt

class Monitor() :
    def __init__(self, n_episodes, policies) :
        self.n_episodes = n_episodes
        self.policies  = policies
        self.trained = False
    
    def train(self) :
        for policy in self.policies :
            policy.train(self.n_episodes)
        return

    def plot_cum_rewards(self) :
        if not self.trained :
            self.train()
        plt.figure()
        for policy in self.policies :
            cumreward  = np.cumsum(policy.rewards)
            plt.plot(range(self.n_episodes), cumreward, label = policy.name)
        plt.legend()
        plt.title('Comparison of different policies.')
        plt.show()
        return
    
    def plot_regret(self) :
        if not self.trained :
            self.train()
        plt.figure()
        for policy in self.policies :
            opt_rewards = policy.env.max_mean * np.ones(self.n_episodes)
            opt_rewards = np.cumsum(opt_rewards)
            cumreward = None
            cumreward  = np.cumsum(policy.rewards)
            plt.plot(range(self.n_episodes), opt_rewards - cumreward, label = policy.name)
        plt.legend()
        plt.title('Comparison of different policies.')
        plt.show()
        return


if __name__ == "__main__":
    env = KarmedBandits(k = 30, std = 5., min_value = 2, max_value = 30)
    u= UCB(c = 1., env = env)
    e = EpsGreedy(eps = 0.1, decay = 1.0, env = env)
    monitor = Monitor(policies = [u, e], n_episodes = 50000)
    monitor.plot_regret()
