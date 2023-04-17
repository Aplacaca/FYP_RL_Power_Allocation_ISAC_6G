import numpy as np
from collections import deque

class ReplayMemoryPool():
    def __init__(self, size=10000):
        self.size = size # Memory size
        self.memories = deque(maxlen=self.size)
    def add_memory(self, state, reward, action, next_state, done, controller_state=None):
        if controller_state is None:
            self.memories.append([state, reward, action, next_state, done])
        else:
            self.memories.append([state, reward, action, next_state, done, controller_state])

    def get_batch(self, batch_size):
        choosen_index = np.random.choice(np.arange(0,len(self.memories)), batch_size, replace=False)
        return [self.memories[i] for i in choosen_index]
