"""
Simple test to check that PPO1 is running with no errors (see issue #50)
"""
from stable_baselines import ppo1


if __name__ == '__main__':
    model = ppo1('MlpPolicy', 'CartPole-v1', schedule='linear', verbose=0)
    model.learn(total_timesteps=1000)
