#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Main test file

Possible candidate environments(Continual action space and continual state space):
LunarLanderContinuous-v2
BipedalWalker-v2
BipedalWalkerHardcore-v2
CarRacing-v0
"""

import numpy as np
import tensorflow as tf

import roboschool
import gym
import sys
import getopt

from stable_baselines.oc.Brain import Brain, Barin_soc

LEARNING_RATE = 1e-3

ALPHA = 0.9
GAUSS_DECAY = 0.9999
BATCH_SIZE = 32

# trace of some variables
SAVE_REWARD_STEP = 200
SAVE_OPTION_STEP = 10000
SAVE_MODEL_STEP = 100
REWARDS_FILE = 'rewards.npy'
DURAS_FILE = 'duras.npy'
OPTIONS_FILE = 'options.npy'


class Comb:

    def __init__(self):

        # test environment
        self.env = gym.make('Fourrooms-v1')
        # self.env = gym.make('LunarLanderContinuous-v2')
        # self.env = gym.make('MountainCarContinuous-v0')
        # self.env = gym.make('Pendulum-v0')

        # seed the env
        self.seed()

        # env_dict
        self.env_dict = {'state_dim': self.env.observation_space.shape[0],
                         'state_scale': self.env.observation_space.high,
                         'action_dim': self.env.action_space.shape[0],
                         'action_scale': self.env.action_space.high}

        # params
        self.params = {'option_num': 4,
                       'epsilon': 0.05,
                       'upper_gamma': 0.9,
                       'lower_gamma': 0.9,
                       'upper_capacity': 100000,
                       'lower_capacity': 100000,
                       'upper_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_policy': LEARNING_RATE,
                       'lower_learning_rate_termin': LEARNING_RATE}

        # RL brain
        self.brain = Brain(env_dict=self.env_dict, params=self.params)

        # some variables
        self.gauss = np.ones(self.params['option_num'])
        self.state = np.zeros(self.env_dict['state_dim'])
        self.target = np.zeros(self.env_dict['state_dim'])
        self.start_state = self.state.copy()
        self.option = -1    # current option
        self.option_reward = 0
        self.option_time = 0

        # the array to record reward and options
        self.rewards = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.duras = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.options = np.zeros(SAVE_OPTION_STEP, dtype=np.int32)
        self.reward_counter = 0             # separate reward
        self.option_counter = 0             # separate option
        self.model_counter = 0              # model counter
        self.model_ordinal = 0              # model ordinal

    def seed(self, seed=42):
        """Wrapper seed the environment for reproducing the result"""

        np.random.seed(seed)
        self.env.seed(np.random.randint(10000))
        tf.random.set_random_seed(np.random.randint(10000))

    def step(self, action):
        """Wrapper for the environment step, normalized state and action"""

        next_state, reward, done, _ = self.env.step(action * self.env_dict['action_scale'])

        return next_state / self.env_dict['state_scale'], reward, done

    def reset(self):
        """Wrapper for the environment step"""

        self.state, self.target = self.env.reset()
        self.state /= self.env_dict['state_scale']

    def update_params(self, params):
        """Update params"""

        self.params.update(params)
        self.brain = Brain(env_dict=self.env_dict, params=self.params)

    def choose_action(self):
        """Choose action with noise"""

        action = self.brain.l_options[self.option].choose_action(self.state)
        action += np.random.normal(0.0, self.gauss[self.option], self.env_dict['action_dim'])
        action = np.clip(action, -1.0, 1.0)

        return action

    def save_model(self):
        """Save model"""

        self.model_counter += 1
        if self.model_counter == SAVE_MODEL_STEP:
            self.model_counter = 0
            self.brain.save_model(self.model_ordinal)
            self.model_ordinal += 1

    def change_option(self):
        """Judge whether change option"""

        self.option_reward = 0
        self.option_time = 0
        self.option = self.brain.u_critic.choose_option(self.state)
        self.start_state = self.state.copy()

    def one_episode(self):
        """Run the process"""

        ep_reward = 0
        t = 0
        self.reset()
        self.change_option()

        while True:

            # choose action with noise
            action = self.choose_action()

            # step the environment
            next_state, reward, done = self.step(action)

            self.brain.l_buffers[self.option].store(self.state, action, reward, next_state)

            # record the option
            self.record_option()

            # accumulate reward
            ep_reward += reward
            t += 1
            self.option_reward += reward
            self.option_time += 1

            # train lower options and decay the variance
            if self.brain.train_option(BATCH_SIZE, self.option):
                self.gauss[self.option] *= GAUSS_DECAY

            # done
            if done:
                print(" Target:{}. End:{}".format(self.target, next_state * self.env_dict['state_scale']))
                print("timestep={},reward={},gauss={}".format(t, ep_reward, self.gauss[self.option]))
                self.record_reward(ep_reward, t)
                break

            # check whether change option
            self.state = next_state

            # if the termination condition satisfied
            if np.random.uniform() < self.brain.l_options[self.option].get_prob(self.state):
                # calculate option reward
                self.option_reward /= self.option_time ** ALPHA
                self.brain.u_buffer.store(self.start_state, self.option, self.option_reward, self.state)
                self.change_option()

        self.brain.train_policy(BATCH_SIZE)
        self.save_model()

    def many_episodes(self, episode_num):
        """Many episodes"""

        if episode_num == 'inf':
            i = 0
            while True:
                print(i, end='')
                i += 1
                self.one_episode()
        elif isinstance(episode_num, int) and episode_num > 0:
            for i in range(episode_num):
                print(i, end='')
                self.one_episode()
        else:
            print('Invalid input! Please input "inf" or a positive integer!')

    def record_option(self):
        """Record option"""

        # record current option
        self.options[self.option_counter] = self.option
        self.option_counter += 1
        # If the buffer is filled, then save it
        if self.option_counter == SAVE_OPTION_STEP:
            self.option_counter = 0
            with open(OPTIONS_FILE, 'ab') as f:
                np.save(f, self.options)

    def record_reward(self, ep_reward, t):
        """Record reward"""

        # record current option
        self.rewards[self.reward_counter] = ep_reward
        self.duras[self.reward_counter] = t
        self.reward_counter += 1
        # If the buffer is filled, then save it
        if self.reward_counter == SAVE_REWARD_STEP:
            self.reward_counter = 0
            with open(REWARDS_FILE, 'ab') as f:
                np.save(f, self.rewards)
            with open(DURAS_FILE, 'ab') as f:
                np.save(f, self.duras)

    def pretrain(self):
        """Pretrain termination"""

        for i in range(self.params['option_num']):
            self.brain.l_options[i].pretrain()


class soft_option_critic(object):
    """ soft option critic"""
    def __init__(self, env):

        # test environment
        self.env = env

        # seed the env
        self.seed()

        # env_dict
        self.env_dict = {'state_dim': self.env.observation_space.shape[0],
                         'state_scale': self.env.observation_space.high,
                         'action_dim': self.env.action_space.shape[0],
                         'action_scale': self.env.action_space.high}

        # params
        self.params = {'option_num': 2,
                       'epsilon': 0.05,
                       'upper_gamma': 0.9,
                       'lower_gamma': 0.9,
                       'cycles': 2,
                       'episode_num': 10000,
                       'max_step': 10000,
                       'warm_start': 1e4,
                       'batch_size': 64,
                       'render': True,
                       'upper_capacity': 100000,
                       'lower_capacity': 1000000,
                       'upper_policy_training': False,
                       'upper_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_critic': LEARNING_RATE,
                       'lower_learning_rate_policy': LEARNING_RATE,
                       'lower_learning_rate_termin': LEARNING_RATE}

        # RL brain
        self.brain = Barin_soc(env_dict=self.env_dict, params=self.params)

        # some variables
        self.gauss = np.ones(self.params['option_num'])
        self.state = np.zeros(self.env_dict['state_dim'])
        self.target = np.zeros(self.env_dict['state_dim'])
        self.start_state = self.state.copy()
        self.option = -1
        self.option_reward = 0
        self.option_time = 0

        # the array to record reward and options
        self.rewards = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.duras = np.zeros(SAVE_REWARD_STEP, dtype=np.float32)
        self.options = np.zeros(SAVE_OPTION_STEP, dtype=np.int32)
        self.reward_counter = 0             # separate reward
        self.option_counter = 0             # separate option
        self.model_counter = 0              # model counter
        self.model_ordinal = 0              # model ordinal

    def train(self):
        """ training process """

        record_reward = np.zeros((self.params['cycles'], self.params['episode_num']), dtype=np.float32)
        total_steps = 0
        for cycle in range(self.params['cycles']):
            for ep in range(self.params['episode_num']):
                ep_reward = 0
                t = 0
                self.reset()
                self.option = self.change_option()

                while True:

                    # choose action with noise
                    action = self.choose_action()

                    # step the environment
                    next_state, reward, done = self.step(action)

                    # record the option
                    self.record_option()

                    # accumulate reward
                    ep_reward += reward
                    t += 1
                    self.option_reward += reward
                    self.option_time += 1

                    if total_steps > self.params['warm_start']:
                        self.brain.training_batch(self.params['batch_size'])

                    # done
                    if done or t >= self.params['max_step']:
                        # print(" Target:{}. End:{}".format(self.target, next_state * self.env_dict['state_scale']))
                        # print(" timestep={}, state={}, action={}, reward={}, option={}".format(total_steps,
                        #                                                                       self.state, action, ep_reward, self.option))
                        print("Done!!!!")
                        break

                    # collect terminaiton function
                    term_prob = self.brain.predict_option_termination(next_state, self.option)

                    # if the termination condition satisfied
                    if term_prob:
                        # change option
                        next_option = self.change_option()
                    else:
                        next_option = self.option

                    self.brain.l_buffer.store(self.state, self.option, action, reward,
                                              next_state, next_option, done)
                    total_steps += 1

                    # check whether change option
                    self.state = next_state
                    self.option = next_option

                record_reward[cycle, ep] = ep_reward
                print('Ep_{}:::Ep_reward_{}'.format(ep, ep_reward))
                # self.save_model()

        return record_reward

    def choose_action(self):
        """Choose action with noise"""

        action = self.brain.choose_action(self.state, self.option)
        # action += np.random.normal(0.0, self.gauss[self.option], self.env_dict['action_dim'])
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action

    def change_option(self):
        """Judge whether change option"""

        option = self.brain.choose_option(self.state, self.params['epsilon'])
        self.start_state = self.state.copy()

        return option

    def seed(self, seed=42):
        """Wrapper seed the environment for reproducing the result"""

        np.random.seed(seed)
        self.env.seed(np.random.randint(10000))
        tf.set_random_seed(np.random.randint(10000))

    def step(self, action):
        """Wrapper for the environment step, normalized state and action"""

        next_state, reward, done, _ = self.env.step(action * self.env_dict['action_scale'])

        return next_state, reward, done

    def reset(self):
        """Wrapper for the environment step"""

        self.state = self.env.reset()
        # self.state /= self.env_dict['state_scale']

    def update_params(self, params):
        """Update params"""

        self.params.update(params)
        self.brain = Brain(env_dict=self.env_dict, params=self.params)

    def save_model(self):
        """Save model"""

        self.model_counter += 1
        if self.model_counter == SAVE_MODEL_STEP:
            self.model_counter = 0
            self.brain.save_model(self.model_ordinal)
            self.model_ordinal += 1

    def record_option(self):
        """Record option"""

        # record current option
        self.options[self.option_counter] = self.option
        self.option_counter += 1
        # If the buffer is filled, then save it
        if self.option_counter == SAVE_OPTION_STEP:
            self.option_counter = 0
            with open(OPTIONS_FILE, 'ab') as f:
                np.save(f, self.options)

    def record_reward(self, ep_reward, t):
        """Record reward"""

        # record current option
        self.rewards[self.reward_counter] = ep_reward
        self.duras[self.reward_counter] = t
        self.reward_counter += 1

        # If the buffer is filled, then save it
        if self.reward_counter == SAVE_REWARD_STEP:
            self.reward_counter = 0
            with open(REWARDS_FILE, 'ab') as f:
                np.save(f, self.rewards)
            with open(DURAS_FILE, 'ab') as f:
                np.save(f, self.duras)

    def pretrain(self):
        """Pretrain termination"""

        for i in range(self.params['option_num']):
            self.brain.l_options[i].pretrain()


def main():
    """Main program"""
    env_list = ['RoboschoolHopper-v1', 'RoboschoolWalker2d-v1',
                'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1',
                'RoboschoolHumanoid-v1']

    env = gym.make('RoboschoolAnt-v1')
    print('action_space_low::', env.action_space.low, 'action_space_high::', env.action_space.high)
    print('state_space_low::', env.observation_space.low, 'state_space_high', env.observation_space.high)

    soc = soft_option_critic(env=env)
    soc.train()


def render():
    """render the network"""

    com = Comb()
    com.brain.restore_model()

    # render lower options
    for i in range(com.params['option_num']):
        com.brain.l_options[i].render()

    # render upper critic
    com.brain.u_critic.render()


if __name__ == '__main__':

    main()
