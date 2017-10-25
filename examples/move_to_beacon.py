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
    for _ in range(_NUM_EPISODES):
        obs = env.reset()

        total_reward = 0
        done = False
        while not done:
            action = move_to_beacon(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print('Episode {} reward: {}'.format(env.episode, total_reward))
    # env.save_replay(env.map_name + '/scripted_example')


def move_to_beacon(obs):
    neutral_y, neutral_x = (obs == _PLAYER_NEUTRAL).nonzero()
    if not neutral_y.any():
        return _NO_OP
    x_coord = int(neutral_x.mean())
    y_coord = int(neutral_y.mean())
    target = np.ravel_multi_index([y_coord, x_coord], obs.shape)
    return target


if __name__ == "__main__":
    main()
