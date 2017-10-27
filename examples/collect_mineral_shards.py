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

    env = gym.make("SC2CollectMineralShards-v0")
    env.settings['visualize'] = False

    episode_reward = np.zeros((_NUM_EPISODES, ))
    for ix in range(_NUM_EPISODES):
        obs = env.reset()

        done = False
        while not done:
            action = collect_mineral_shards(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward[ix] += reward

        print('Episode {} reward: {}'.format(env.episode, episode_reward[ix]))

    print('Average reward for {} episodes: {}'.format(_NUM_EPISODES, np.mean(episode_reward)))


def collect_mineral_shards(obs):
    neutral_y, neutral_x, _ = (obs == _PLAYER_NEUTRAL).nonzero()
    player_y, player_x, _ = (obs== _PLAYER_FRIENDLY).nonzero()
    if not neutral_y.any() or not player_y.any():
        return _NO_OP
    player = [int(player_x.mean()), int(player_y.mean())]
    shards = np.array(list(zip(neutral_x, neutral_y)))
    closest_ix = np.argmin(np.linalg.norm(np.array(player) - shards, axis=1))
    target = np.ravel_multi_index(shards[closest_ix], obs.shape[:2])
    return target


if __name__ == "__main__":
    main()
