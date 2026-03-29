import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, T):
        self.T = T
        self.t = 0

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, X):
        pass


class LCBAgent(Agent):
    def __init__(self, T):
        super().__init__(T)
        self.avg_rewards = np.zeros(T)
        self.N_pulls = np.arange(T)

    def pull_arm(self):
        if self.t < 2:
            a = 0
        else:
            lcbs = self.avg_rewards[:self.t] - np.sqrt(6*np.log(self.T)/(self.t-self.N_pulls[:self.t]))
            a = np.argmax(lcbs)
        return a
    
    def update(self, X):
        self.t += 1
        self.avg_rewards[:self.t] +=  (X-self.avg_rewards[:self.t])/(self.t-self.N_pulls[:self.t])

class UCBAgent(Agent):
    def __init__(self, T):
        super().__init__(T)
        self.avg_rewards = np.zeros(T)
        self.N_pulls = np.arange(T)
        self.buffer = 50

    def pull_arm(self):
        if self.t < self.buffer + 1:
            a = 0
        else:
            lcbs = self.avg_rewards[:(self.t-self.buffer)] + np.sqrt(6*np.log(self.T)/(self.t-self.N_pulls[:(self.t-self.buffer)]))
            a = np.argmax(lcbs)
        return a
    
    def update(self, X):
        self.t += 1
        self.avg_rewards[:self.t] +=  (X-self.avg_rewards[:self.t])/(self.t-self.N_pulls[:self.t])

