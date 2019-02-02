import numpy as np

class KarmedBandits():
    def __init__(self, k, std, min_value, max_value):
        self.k = k
        self.std = std
        self.means = np.random.randint(min_value, max_value, size = self.k)
        self.best_arm = np.argmax(self.means)
        self.max_mean = np.max(self.means)
        return

    def reward(self, i) :
        assert i < self.k
        return np.random.normal(size = 1, loc = self.means[i], scale = self.std)










