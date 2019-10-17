#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The upper policy and the whole training process defined here
"""
import numpy as np
import tensorflow as tf

from .Option import Option, Soft_option
from .LCritic import LCritic, Soft_critic
from .Buffer import Buffer, Buffer_option
from .UCritic import UCritic

# the number of models that will be saved
MODEL_NUM = 100
DELAY = 10000


class Brain(object):

    def __init__(self, env_dict, params):
        """
        option_num, state_dim, action_dim, action_bound, gamma, learning_rate, replacement,
                 buffer_capacity, epsilon
        gamma: (u_gamma, l_gamma)
        learning_rate: (lr_u_policy, lr_u_critic, lr_option, lr_termin, lr_l_critic)
        """

        # session
        self.sess = tf.Session()

        # environment parameters
        self.sd = env_dict['state_dim']
        self.ad = env_dict['action_dim']
        a_bound = env_dict['action_scale']
        assert a_bound.shape == (self.ad,), 'Action bound does not match action dimension!'

        # hyper parameters
        self.on = params['option_num']
        epsilon = params['epsilon']
        u_gamma = params['upper_gamma']
        l_gamma = params['lower_gamma']
        u_capac = params['upper_capacity']
        l_capac = params['lower_capacity']
        u_lrcri = params['upper_learning_rate_critic']
        l_lrcri = params['lower_learning_rate_critic']
        l_lrpol = params['lower_learning_rate_policy']
        l_lrter = params['lower_learning_rate_termin']

        # Upper critic and buffer
        self.u_critic = UCritic(session=self.sess, state_dim=self.sd, option_num=self.on,
                                gamma=u_gamma, epsilon=epsilon, learning_rate=u_lrcri)

        self.u_buffer = Buffer(state_dim=self.sd, action_dim=1, capacity=u_capac)

        # Lower critic, options and buffer HER
        self.l_critic = LCritic(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                gamma=l_gamma, learning_rate=l_lrcri)

        """options and buffers"""
        self.l_options = [Option(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                 ordinal=i, learning_rate=[l_lrpol, l_lrter])
                          for i in range(self.on)]
        self.l_buffers = [Buffer(state_dim=self.sd, action_dim=self.ad, capacity=l_capac)
                          for i in range(self.on)]

        # Initialize all coefficients and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=MODEL_NUM)

        # counter for training termination
        self.tc = 0

    def train_policy(self, batch_size):
        """Train upper critic(policy)"""

        if self.u_buffer.pointer > batch_size:

            # sample batches
            state_batch, option_batch, reward_batch, next_state_batch, _ = self.u_buffer.sample(batch_size)

            # training
            self.u_critic.train(state_batch, option_batch, reward_batch, next_state_batch)

    def train_option(self, batch_size, option):
        """Train option and l_critic"""

        if self.l_buffers[option].pointer > batch_size:

            # sample batches
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = \
                self.l_buffers[option].sample(batch_size)

            next_action_batch = self.l_options[option].get_target_actions(next_state_batch)

            # train lower critic
            self.l_critic.train(state_batch, action_batch, reward_batch, next_state_batch,
                                next_action_batch, terminal_batch)

            # get affiliated batch
            q_gradients_batch = self.l_critic.q_gradients(state_batch, action_batch)

            if self.tc == DELAY:
                advantage_batch = self.l_critic.q_batch(state_batch, action_batch) - \
                                  self._value_batch(state_batch)
                self.tc = 0
            else:
                advantage_batch = None
                self.tc += 1

            # train lower options
            self.l_options[option].train(state_batch, q_gradients_batch, advantage_batch)

            return True

        return False

    def save_model(self, model_name, step):
        """Save current model"""

        self.saver.save(self.sess, './model/' + model_name + '.ckpt', global_step=step, write_meta_graph=True)

    def restore_model(self, model_name):
        """Restore trained model"""

        ckpt = tf.train.get_checkpoint_state('./model/' + model_name)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def _value_batch(self, state_batch):
        """The upper policy average of Q value for each option
        :return: the value
        """

        batch_size = state_batch.shape[0]
        value_batch = np.zeros((batch_size, 1))
        action_batch = [self.l_options[i].get_actions(state_batch) for i in range(self.on)]
        q_batch = [self.l_critic.q_batch(state_batch, action_batch[i]) for i in range(self.on)]
        distribution_batch = self.u_critic.get_distribution(state_batch)

        # calculate the value function
        for i in range(batch_size):
            for j in range(self.on):
                value_batch[i] += q_batch[j][i] * distribution_batch[i, j]

        return value_batch


class Barin_soc(object):
    def __init__(self, env_dict, params):
        """
        option_num, state_dim, action_dim, action_bound, gamma, learning_rate, replacement,
                 buffer_capacity, epsilon
        gamma: (u_gamma, l_gamma)
        learning_rate: (lr_u_policy, lr_u_critic, lr_option, lr_termin, lr_l_critic)
        """

        # session
        self.sess = tf.Session()

        # environment parameters
        self.sd = env_dict['state_dim']
        self.ad = env_dict['action_dim']
        self.on = params['option_num']
        a_bound = env_dict['action_scale']
        assert a_bound.shape == (self.ad,), 'Action bound does not match action dimension!'

        # hyper parameters
        epsilon = params['epsilon']
        u_gamma = params['upper_gamma']
        l_gamma = params['lower_gamma']
        u_capac = params['upper_capacity']
        l_capac = params['lower_capacity']
        u_lrcri = params['upper_learning_rate_critic']
        l_lrcri = params['lower_learning_rate_critic']
        l_lrpol = params['lower_learning_rate_policy']
        l_lrter = params['lower_learning_rate_termin']
        upper_policy_training = params['upper_policy_training']

        # Upper critic and buffer
        self.u_critic = UCritic(session=self.sess, state_dim=self.sd, option_num=self.on,
                                gamma=u_gamma, epsilon=epsilon, learning_rate=u_lrcri)

        self.u_buffer = Buffer(state_dim=self.sd, action_dim=1, capacity=u_capac)

        # Lower critic, options and buffer HER
        self.critics = Soft_critic(session=self.sess, state_dim=self.sd, action_dim=self.ad, option_dim=self.on,
                                gamma=l_gamma, learning_rate=l_lrcri)

        # options and buffers
        self.options = [Soft_option(session=self.sess, state_dim=self.sd, action_dim=self.ad,
                                   option_name=str(i), learning_rate=[l_lrpol, l_lrter]) for i in range(self.on)]

        self.l_buffer = Buffer_option(state_dim=self.sd, action_dim=self.ad, capacity=l_capac)

        # Initialize all coefficients and saver
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=MODEL_NUM)

        # counter for training termination
        self.tc = 0

    def train_option(self, batch_size, state_batch, option_batch,
                    next_state_batch, terminal_batch, q_value_batch, adv_value_batch):
        """Train option"""

        for i in range(batch_size):
            term_prob_next = self.options[int(option_batch[i][0])].get_term_prob(next_state_batch[i])
            # print('q_value_batch', q_value_batch[i])
            # print('adv_batch', adv_value_batch[i])
            # print('term_prob_next', term_prob_next)
            _ = self.options[int(option_batch[i][0])].train_option(
                [state_batch[i]],
                [[q_value_batch[i]]],
                term_prob_next[np.newaxis, :],
                terminal_batch[i][np.newaxis, :],
                [[adv_value_batch[i]]])

    def train_critic(self, state_batch, option_batch, action_batch, reward_batch, next_state_batch,
                     next_option_batch, terminal_batch, log_policy_batch, policy_batch):
        """Train critic network"""

        # print('state_batch', state_batch)
        # print('option_batch', option_batch)
        # print('action_batch', action_batch)
        # print('log_policy_batch', log_policy_batch)
        # print('termination_batch', terminal_batch)
        self.critics.train_critic(state_batch,
                                  option_batch,
                                  action_batch,
                                  reward_batch,
                                  next_state_batch,
                                  next_option_batch,
                                  terminal_batch,
                                  log_policy_batch,
                                  policy_batch
                                  )

    def training_batch(self, batch_size):
        # sample batches
        state_batch, option_batch, action_batch, reward_batch, next_state_batch, next_option_batch, terminal_batch = \
            self.l_buffer.sample(batch_size)

        # print('next_batch', next_state_batch, 'next_option_batch', next_option_batch)

        # # calculate state-option-action value
        log_policy_batch = np.zeros([batch_size, 1], dtype=np.float32)
        policy_batch = np.zeros([batch_size, self.ad], dtype=np.float32)

        for i in range(batch_size):
            _, policy_batch[i], log_policy_batch[i] = self.options[int(option_batch[i][0])].get_actions(state_batch[i])

        q_value_batch = self.critics.state_option_action_value_batch(state_batch, policy_batch, option_batch)

        # on-policy and off-policy
        adv_value_batch = self.critics.advantage_batch(next_state_batch, option_batch)

        self.train_option(batch_size, state_batch, option_batch,
                     next_state_batch, terminal_batch, q_value_batch, adv_value_batch)

        self.train_critic(state_batch, option_batch, action_batch, reward_batch, next_state_batch,
                          next_option_batch, terminal_batch, log_policy_batch, policy_batch)

    def choose_action(self, state, option):
        """choose action"""

        return self.options[option].choose_action(state)

    def choose_option(self, state, eps):
        """choose option"""

        if np.random.rand() < eps:
            option = np.random.choice(self.on)
        else:
            option = self.critics.predict_state_value(state)[0].argmax()

        return option

    def predict_option_termination(self, state, option):
        """predict termination"""

        termination_probs = self.options[option].get_term_prob(state)
        return termination_probs

    def save_model(self, model_name, step):
        """Save current model"""

        self.saver.save(self.sess, './model/' + model_name + '.ckpt', global_step=step, write_meta_graph=True)

    def restore_model(self, model_name):
        """Restore trained model"""

        ckpt = tf.train.get_checkpoint_state('./model/' + model_name)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def train_option_policy(self, batch_size):
        """Train upper critic(policy)"""

        if self.u_buffer.pointer > batch_size:

            # sample batches
            state_batch, option_batch, reward_batch, next_state_batch, _ = self.u_buffer.sample(batch_size)

            # training
            self.u_critic.train(state_batch, option_batch, reward_batch, next_state_batch)
