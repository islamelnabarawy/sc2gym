import sys

import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2MoveToBeacon-v1")
    obs = env.reset()

    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print('Episode {} reward: {}'.format(env.episode, total_reward))
    # env.save_replay(env.map_name)


if __name__ == "__main__":
    main()
