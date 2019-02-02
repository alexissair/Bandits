import numpy as np
from Bandits import KarmedBandits

class UCB() :
    def __init__(self, c, env) :
        self.env = env
        self.scores = np.zeros(self.env.k)
        self.c = c
        self.name = 'UCB'
        self.rewards = []
        self.actions = []
        self.calls = np.zeros(self.env.k)
        self.mean_reward = np.zeros(self.env.k)
        return

    def pick_action(self):
        if len(self.actions) < self.env.k :
            act = len(self.actions)
        else :
            act = np.argmax(self.scores)
        return act

    def step(self) :
        act = self.pick_action()
        rew = self.env.reward(act)
        if self.calls[act] == 0 :
            self.mean_reward[act] = rew
        else :
            self.mean_reward[act] = 1/self.calls[act]*rew + (self.calls[act] - 1)/self.calls[act] * self.mean_reward[act]
        self.scores = self.mean_reward + self.c * np.sqrt(np.log(len(self.actions))/(self.calls + 1e-15))
        self.rewards.append(rew)
        self.actions.append(act)
        return

    def train(self, n_episodes) :
        for _ in range(n_episodes) :
            self.step()
        return




if __name__ == "__main__":
    env = KarmedBandits(k = 10, std = 4., min_value = 0, max_value = 50)
    u= UCB(c = 30., env = env)
    u.train(100)


