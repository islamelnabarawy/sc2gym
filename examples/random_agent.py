import sys

import numpy as np
import gym
from absl import flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

from pysc2.lib import actions, features

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


def main():
    FLAGS(sys.argv)

    env = gym.make("SC2Game-v0")
    env.settings['map_name'] = 'CollectMineralShards'
    obs = env.reset()

    done = False
    while not done:
        action = random_action(env, obs)
        obs, reward, done, _ = env.step(action)

    env.close()


def random_action(env, obs):
    function_id = np.random.choice(obs.observation["available_actions"])
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in env.action_spec.functions[function_id].args]
    return [function_id] + args


if __name__ == "__main__":
    main()
