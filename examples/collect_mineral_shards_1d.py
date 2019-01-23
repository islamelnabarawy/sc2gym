"""
   Copyright 2017 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import argparse

import numpy as np

from examples.base_example import BaseExample

__author__ = 'Islam Elnabarawy'
__description__ = 'Run a scripted example using the SC2CollectMineralShards-v0 environment.'

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2CollectMineralShards-v0"


class CollectMineralShards1d(BaseExample):
    def __init__(self, visualize=False, step_mul=None, random_seed=None) -> None:
        super().__init__(_ENV_NAME, visualize, step_mul, random_seed)

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        player_y, player_x = (obs[0] == _PLAYER_FRIENDLY).nonzero()
        if not neutral_y.any():
            raise Exception('No minerals found!')
        if not player_y.any():
            raise Exception('No marines found!')
        player = [np.ceil(player_x.mean()).astype(int), np.ceil(player_y.mean()).astype(int)]
        shards = np.array(list(zip(neutral_x, neutral_y)))
        closest_ix = np.argmin(np.linalg.norm(np.array(player) - shards, axis=1))
        target = np.ravel_multi_index(shards[closest_ix], obs.shape[1:])
        return target


def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--visualize', type=bool, default=False,
                        help='show the pysc2 visualizer')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='number of episodes to run')
    parser.add_argument('--step-mul', type=int, default=None,
                        help='number of game steps to take per turn')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='the random seed to pass to the game environment')
    args = parser.parse_args()

    example = CollectMineralShards1d(args.visualize, args.step_mul, args.random_seed)
    rewards = example.run(args.num_episodes)
    if rewards:
        print('Total reward: {}'.format(rewards.sum()))
        print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
        print('Minimum reward: {}'.format(rewards.min()))
        print('Maximum reward: {}'.format(rewards.max()))


if __name__ == "__main__":
    main()
