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

    env = gym.make("SC2Game-v0")
    env.settings['map_name'] = 'BuildMarines'
    env.settings['visualize'] = False

    obs = env.reset()

    done = False
    while not done:
        action = _NO_OP
        obs, reward, done, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
