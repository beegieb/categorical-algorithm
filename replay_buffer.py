"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, input_size, observation_window=3, random_seed=None):

        self.buffer_size = buffer_size
        self.actions = np.empty(buffer_size, dtype=np.uint8)
        self.rewards = np.empty(buffer_size, dtype=np.integer)
        size = [buffer_size] + list(input_size)
        self.screens = np.empty(size, dtype=np.float16)
        self.terminals = np.empty(buffer_size, dtype=np.bool)
        self.observation_window = observation_window
        self.dims = tuple(input_size)
        self.count = 0
        self.current = 0
        self.total_count = 0

    def add(self, screen, action, reward, terminal):
        assert screen.shape == tuple(self.dims)
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size
        self.total_count = self.total_count + 1

    def getState(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.observation_window - 1:
            # use faster slicing
            return self.screens[(index - (self.observation_window - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.observation_window))]
            return self.screens[indexes, ...]

    def sample_batch(self, batch_size):
        # pre-allocate prestates and poststates for minibatch
        prestates = np.empty((batch_size, self.observation_window) + self.dims[:2], dtype=np.float16)
        poststates = np.empty((batch_size, self.observation_window) + self.dims[:2], dtype=np.float16)

        # memory must include poststate, prestate and history
        assert self.count > self.observation_window
        # sample random indexes
        indexes = []
        while len(indexes) < batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.observation_window, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current > index - self.observation_window:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.observation_window):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            prestates[len(indexes), ...] = self.getState(index - 1)[..., 0]
            poststates[len(indexes), ...] = self.getState(index)[..., 0]
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return (np.transpose(prestates, (0, 2, 3, 1)),
                actions,
                rewards,
                np.transpose(poststates, (0, 2, 3, 1)),
                terminals)

    def save(self, checkpoint_dir):
        pass
        # for idx, (name, array) in enumerate(
        #         zip(['actions', 'rewards', 'screens', 'terminals'],
        #             [self.actions, self.rewards, self.screens, self.terminals])):
        #     # save_npy(array, os.path.join(checkpoint_dir, name))

    def load(self, checkpoint_dir):
        pass
        # for idx, (name, array) in enumerate(
        #         zip(['actions', 'rewards', 'screens', 'terminals'],
        #             [self.actions, self.rewards, self.screens, self.terminals])):
        #     array = load_npy(os.path.join(checkpoint_dir, name))
