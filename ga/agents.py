import numpy as np


# Class that contains the neural net
class DummyAgent:
    def __init__(self, total_lockdown_days, lockdown_length, sim_length):
        self.lockdown_length = lockdown_length
        self.sim_length = sim_length
        self.num_lockdowns = int(total_lockdown_days/lockdown_length)
        self.total_lockdown_days = total_lockdown_days
        self.params = (np.random.random(self.num_lockdowns)*0).astype(float)
        np.sort(self.params)

    def forward(self, i):
        for elem in self.params:
            if i > np.floor(elem) and i <= np.floor(elem + self.lockdown_length):
                return 1
        return 0
