import sys

import numpy as np
import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS

_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_NUM_EPISODES = 10


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2MoveToBeacon-v0")
    env.settings['visualize'] = False

    for ix in range(_NUM_EPISODES):
        obs = env.reset()

        done = False
        while not done:
            action = move_to_beacon(obs)
            obs, reward, done, _ = env.step(action)


def move_to_beacon(obs):
    neutral_y, neutral_x, _ = (obs == _PLAYER_NEUTRAL).nonzero()
    if not neutral_y.any():
        return _NO_OP
    target = [int(neutral_x.mean()), int(neutral_y.mean())]
    target = np.ravel_multi_index(target, obs.shape[:2])
    return target


if __name__ == "__main__":
    main()
