import gym

from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC


# Custom MLP policy of three layers of size 128 each
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")


# Create and wrap the environment
env = gym.make('Acrobot-v1')

print('action_space_low::', env.action_space, 'action_space_high::', env.action_space)
print('state_space_low::', env.observation_space.low, 'state_space_high', env.observation_space.high)

env = DummyVecEnv([lambda: env])

# model = SAC(CustomSACPolicy, env, verbose=1)

# Train the agent
# model.learn(total_timesteps=100000)
