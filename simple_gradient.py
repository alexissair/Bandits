import numpy as np
from epsgreedy import *

class SimpleGradient() :
    def __init__(self, env, eps, decay) :
        self.env = env
        self.eps = eps
        self.decay = decay
        self.scores = np.zeros(self.env.k)
        self.name = 'Simple Gradient'
        self.rewards = []
        self.actions = []
        self.calls = np.zeros(self.env.k)
        return
    
    def pick_action(self):
        u = np.random.rand()
        if u <= self.eps :
            p = np.random.randint(self.env.k)
        else :
            p = np.argmax(self.scores)
        return p
    
    def step(self) :
        act = self.pick_action()
        rew = self.env.reward(act)
        self.calls[act] += 1
        self.scores[act] += 1/self.calls[act] * (rew - self.scores[act])
        self.rewards.append(rew)
        self.actions.append(act)
        return


    def train(self, n_episodes) :
        for _ in range(n_episodes) :
            self.step()
        return
    

        
if __name__ == "__main__":
    env = KarmedBandits(k = 10, std = 4., min_value = 2, max_value = 30)
    e = SimpleGradient(eps = 0.1, decay = 1., env = env)
    e.train(100)
    print(e.rewards)