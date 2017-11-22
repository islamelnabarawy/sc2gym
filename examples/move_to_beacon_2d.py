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
from examples.base_example import BaseExample

__author__ = 'Islam Elnabarawy'


_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2MoveToBeacon-v1"
_VISUALIZE = False
_STEP_MUL = None
_NUM_EPISODES = 10


class MoveToBeacon2d(BaseExample):
    def __init__(self, visualize=False, step_mul=None) -> None:
        super().__init__(_ENV_NAME, visualize, step_mul)

    def get_action(self, env, obs):
        neutral_y, neutral_x = (obs[0] == _PLAYER_NEUTRAL).nonzero()
        if not neutral_y.any():
            raise Exception("Beacon not found!")
        target = [int(neutral_x.mean()), int(neutral_y.mean())]
        return target


def main():
    example = MoveToBeacon2d(_VISUALIZE, _STEP_MUL)
    rewards = example.run(_NUM_EPISODES)
    print('Total reward: {}'.format(rewards.sum()))
    print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    print('Minimum reward: {}'.format(rewards.min()))
    print('Maximum reward: {}'.format(rewards.max()))


if __name__ == "__main__":
    main()
