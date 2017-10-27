import sys

import gym
from absl import flags
from pysc2.lib import actions

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2BuildMarines-v0")
    obs = env.reset()

    total_reward = 0
    done = False
    while not done:
        action = _NO_OP
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print('Episode {} reward: {}'.format(env.episode, total_reward))
    # env.save_replay(env.map_name)


if __name__ == "__main__":
    main()
