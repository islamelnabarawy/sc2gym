import numpy as np

from examples.base_example import BaseExample

__author__ = 'Islam Elnabarawy'

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2CollectMineralShards-v0"
_VISUALIZE = False
_STEP_MUL = None
_NUM_EPISODES = 10


class CollectMineralShards1d(BaseExample):
    def __init__(self, visualize=False, step_mul=None) -> None:
        super().__init__(_ENV_NAME, visualize, step_mul)

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
    example = CollectMineralShards1d(_VISUALIZE, _STEP_MUL)
    rewards = example.run(_NUM_EPISODES)
    print('Total reward: {}'.format(rewards.sum()))
    print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    print('Minimum reward: {}'.format(rewards.min()))
    print('Maximum reward: {}'.format(rewards.max()))


if __name__ == "__main__":
    main()
