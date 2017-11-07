import sys

import numpy as np
import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_NUM_EPISODES = 10


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2CollectMineralShards-v1")
    env.settings['visualize'] = False

    for ix in range(_NUM_EPISODES):
        obs = env.reset()

        done = False
        while not done:
            action = collect_mineral_shards(obs)
            obs, reward, done, _ = env.step(action)

    env.close()


def collect_mineral_shards(obs):
    neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
    player_y, player_x = (obs[0] == _PLAYER_FRIENDLY).nonzero()
    if not neutral_y.any() or not player_y.any():
        return _NO_OP
    player = [np.ceil(player_x.mean()).astype(int), np.ceil(player_y.mean()).astype(int)]
    shards = np.array(list(zip(neutral_x, neutral_y)))
    closest_ix = np.argmin(np.linalg.norm(np.array(player) - shards, axis=1))
    target = shards[closest_ix]
    return target


if __name__ == "__main__":
    main()
