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

_NUM_EPISODES = 10

_NEXT_GROUP = 0


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2CollectMineralShards-v2")
    env.settings['visualize'] = False

    for ix in range(_NUM_EPISODES):
        obs = env.reset()

        done = False
        while not done:
            action = collect_mineral_shards(obs)
            obs, reward, done, _ = env.step(action)

    env.close()


def collect_mineral_shards(obs):
    # HACK: Using global variable, to be replaced by defining a class instead of a function
    global _NEXT_GROUP
    neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
    marine_y, marine_x = ((obs[0] == _PLAYER_FRIENDLY) - obs[1]).nonzero()
    if not neutral_y.any():
        raise Exception('No minerals found!')
    if not marine_y.any():
        marine_y, marine_x, _ = obs[1].nonzero()
    if not marine_y.any():
        raise Exception('No marines found!')
    marine = [np.ceil(marine_x.mean()).astype(int), np.ceil(marine_y.mean()).astype(int)]
    shards = np.array(list(zip(neutral_x, neutral_y)))
    closest_ix = np.argmin(np.linalg.norm(np.array(marine) - shards, axis=1))
    target = shards[closest_ix].tolist()
    group = [_NEXT_GROUP+1]
    _NEXT_GROUP = (_NEXT_GROUP + 1) % 2
    return group + target


if __name__ == "__main__":
    main()
