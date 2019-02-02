from Bandits import *


class EpsGreedy():
    def __init__(self, eps, decay, env) :
        self.eps = eps
        self.decay = decay
        self.env = env
        self.name = 'Epsilon Greedy'
        self.rewards = []
        self.actions = []
        self.calls = np.zeros(self.env.k)
        self.mean_reward = np.zeros(self.env.k)
        return

    def pick_action(self):
        u = np.random.rand()
        if u <= self.eps :
            p = np.random.randint(self.env.k)
        else :
            p = np.argmax(self.mean_reward)
        return p

    def step(self) :
        act = self.pick_action()
        rew = self.env.reward(act)
        if self.calls[act] == 0 :
            self.mean_reward[act] = rew
        else :
            self.mean_reward[act] = 1/self.calls[act]*rew + (self.calls[act] - 1)/self.calls[act] * self.mean_reward[act]
        self.calls[act] += 1
        self.eps *= self.decay
        self.rewards.append(rew)
        self.actions.append(act)
        return

    def train(self, n_episodes) :
        for _ in range(n_episodes) :
            self.step()
        return



if __name__ == "__main__":
    env = KarmedBandits(k = 10, std = 4.)
    e = EpsGreedy(eps = 0.9, decay = 0.95, env = env)
    e.train(100)
    print(e.mean_reward)
