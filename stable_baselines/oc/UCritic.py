#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
An implementation of upper critic, using DQN
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

LAYER1 = 32

EVAL_SCOPE = 'eval_upper_critic'
TARGET_SCOPE = 'target_upper_critic'

INIT_WEIGHT = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
INIT_BIAS = tf.constant_initializer(0.1)


class UCritic:

    def __init__(self, session, state_dim, option_num, gamma, epsilon, tau=1e-2, learning_rate=1e-3):
        """Initiate the critic network for normalized states and options"""

        # tensorflow session
        self.sess = session

        # environment parameters
        self.sd = state_dim
        self.on = option_num
        self.eps = epsilon

        # some placeholder
        self.r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='upper_reward')
        self.o = tf.placeholder(dtype=tf.int32, shape=(None, 1), name='option')

        # evaluation and target network
        self.s, self.q = self._q_net(scope=EVAL_SCOPE, trainable=True)
        self.s_, q_ = self._q_net(scope=TARGET_SCOPE, trainable=False)

        # index the q and get max q_
        qo = tf.gather_nd(params=self.q,
                          indices=tf.stack([tf.range(tf.shape(self.o)[0], dtype=tf.int32),
                                            self.o[:, 0]], axis=1))
        qm = self.r + gamma * tf.reduce_max(q_, axis=1)

        # soft update
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=EVAL_SCOPE)
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TARGET_SCOPE)
        self.update = [tf.assign(t, t + tau * (e - t)) for t, e in zip(target_params, eval_params)]

        # define the error and optimizer
        self.loss = tf.losses.mean_squared_error(labels=qm[:, 0], predictions=qo)
        self.op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=eval_params)

        # pretrain placeholder
        self.tru = tf.placeholder(dtype=tf.int32, shape=(None, self.on))
        self.lo = tf.losses.hinge_loss(labels=self.tru, logits=self.q)
        self.pop = tf.train.AdamOptimizer(1e-3).minimize(self.lo)

    def train(self, state_batch, option_batch, reward_batch, next_state_batch):
        """Train the critic network"""

        # minimize the loss
        self.sess.run(self.op, feed_dict={
            self.s: state_batch,
            self.o: option_batch,
            self.r: reward_batch,
            self.s_: next_state_batch
        })

        # target update
        self.sess.run(self.update)

    def choose_option(self, state):
        """Get the q batch"""

        if np.random.rand() < self.eps:
            return np.random.choice(self.on)
        else:
            return self.sess.run(self.q, feed_dict={
                self.s: state[np.newaxis, :]
            })[0].argmax()

    def get_distribution(self, state_batch):
        """Get the option distribution"""

        # get q batch
        res = self.sess.run(self.q, feed_dict={
            self.s: state_batch
        })
        index = np.argmax(res, axis=1)
        res = np.ones(res.shape) * self.eps / self.on
        res[np.arange(res.shape[0], dtype=np.int32), index] += 1 - self.eps

        return res

    def pretrain(self):
        """Pretrain upper critic"""

        num = 50
        delta = 2.0 / (num - 1)
        test_state = -np.ones((num * num, 2))
        test_label = np.zeros((num * num, 4), dtype=np.int32)
        labels = np.eye(4, dtype=np.int32)
        for i in range(num):
            for j in range(num):
                o = i * num + j
                s = np.array([i * delta, j * delta])
                test_state[o] += s
                if test_state[o, 0] > 0 and test_state[o, 1] > 0:
                    test_label[o] = labels[0]
                elif test_state[o, 0] < 0 < test_state[o, 1]:
                    test_label[o] = labels[1]
                elif test_state[o, 0] < 0 and test_state[o, 1] < 0:
                    test_label[o] = labels[2]
                elif test_state[o, 1] < 0 < test_state[o, 0]:
                    test_label[o] = labels[3]

        while True:
            self.sess.run(self.pop, feed_dict={
                self.s: test_state,
                self.tru: test_label
            })
            a = self.sess.run(self.lo, feed_dict={
                self.s: test_state,
                self.tru: test_label
            })
            if a < 1e-3:
                break

    def _q_net(self, scope, trainable):
        """Generate evaluation/target q network"""

        with tf.variable_scope(scope):

            state = tf.placeholder(dtype=tf.float32, shape=(None, self.sd), name='state')

            x = tf.layers.dense(state, LAYER1, activation=tf.nn.relu,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='dense1')

            q = tf.layers.dense(x, self.on, activation=None,
                                kernel_initializer=INIT_WEIGHT, bias_initializer=INIT_BIAS,
                                trainable=trainable, name='layer2')

        return state, q

    def render(self):
        """Render the critic network"""

        if self.sd == 2:
            num = 50
            X = np.linspace(-1.0, 1.0, num)
            Y = np.linspace(-1.0, 1.0, num)
            C = np.zeros((num, num))

            s = np.zeros((num, 2))
            for i in range(num):
                s[:, 0] = X[i]
                s[:, 1] = Y
                C[:, i] = self.sess.run(self.q, feed_dict={
                    self.s: s
                }).argmax(axis=1)
            im = plt.pcolor(X, Y, C, cmap='jet')
            plt.title('critic output')
            plt.colorbar(im)
            plt.show()
