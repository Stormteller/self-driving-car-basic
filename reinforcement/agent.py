import random

import numpy as np
from collections import deque

from .dqn_model import DqnModel


class Agent:
    def __init__(self, state_shape, action_size):
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.action_size = action_size

        self.model = DqnModel().build_model(state_shape, action_size)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state.get('image'))
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        sample = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in sample:
            next_state_rewards = self.model.predict(next_state.get('image'))
            target = reward + self.gamma * np.amax(next_state_rewards)
            target_f = self.model.predict(state.get('image'))
            target_f[0][action] = target
            self.model.fit(state.image, target_f, nb_epoch=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


