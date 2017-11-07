import numpy as np
import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS
FLAGS([__file__])


class BaseExample(object):
    def __init__(self, env_name, visualize=False, step_mul=None) -> None:
        super().__init__()
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul

    def run(self, num_episodes=1):
        env = gym.make(self.env_name)
        env.settings['visualize'] = self.visualize
        env.settings['step_mul'] = self.step_mul

        episode_rewards = np.zeros((num_episodes, ), dtype=np.int32)
        for ix in range(num_episodes):
            obs = env.reset()

            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, reward, done, _ = env.step(action)

            episode_rewards[ix] = env.episode_reward

        env.close()

        return episode_rewards

    def get_action(self, env, obs):
        raise NotImplementedError('Inherited classes must override get_action() method')
