#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of random replay buffer
"""

import numpy as np


class Buffer:

    def __init__(self, state_dim, action_dim, capacity):

        self.sd = state_dim
        self.ad = action_dim
        self.capacity = capacity
        self.buffer = np.zeros(shape=(capacity, state_dim * 2 + action_dim + 1 + 1))
        self.isFilled = False
        self.pointer = 0

    def store(self, state, action, reward, next_state, terminal):
        """Store the transition"""

        # flatten the transition tuple
        transition = np.hstack((state, action, reward, next_state, terminal))
        self.buffer[self.pointer, :] = transition
        self.pointer += 1

        # prevent overflow
        if self.pointer == self.capacity:
            self.isFilled = True
            self.pointer = 0

    def sample(self, num):
        """Sample batch"""

        # sample batch memory from all memory
        index = np.random.choice(self.pointer-2, size=num)
        batch = self.buffer[index, :]

        # separate the batch memory to different dimensions
        sa = self.sd + self.ad
        return batch[:, :self.sd], \
               batch[:, self.sd: sa], \
               batch[:, sa: sa + 1], \
               batch[:, sa + 1: -1], \
               batch[:, -1:0]


class Buffer_option(object):
    def __init__(self, state_dim, action_dim, capacity):
        """ store option """

        self.sd = state_dim
        self.ad = action_dim
        self.capacity = capacity

        # state, option, action, reward, next_state, termination
        self.buffer = np.zeros(shape=(capacity, state_dim * 2 + action_dim + 1 + 2 + 1))
        self.isFilled = False
        self.pointer = 0

    def store(self, state, option, action, reward, next_state, next_option, terminal):
        """Store the transition"""

        # flatten the transition tuple
        transition = np.hstack((state, option, action, reward, next_state, next_option, terminal))
        # print('trasition', transition)
        self.buffer[self.pointer, :] = transition
        self.pointer += 1

        # prevent overflow
        if self.pointer == self.capacity:
            self.isFilled = True
            self.pointer = 0

    def sample(self, num):
        """Sample batch"""

        # sample batch memory from all memory
        index = np.random.choice(self.pointer-2, size=num)
        batch = self.buffer[index, :]

        # separate the batch memory to different dimensions
        sa = self.sd + self.ad + 1
        return batch[:, :self.sd], \
               batch[:, self.sd: self.sd+1], \
               batch[:, self.sd + 1: sa], \
               batch[:, sa: sa + 1], \
               batch[:, sa + 1: sa + 1 + self.sd], \
               batch[:, sa + 1 + self.sd: -1], \
               batch[:, -1:]
