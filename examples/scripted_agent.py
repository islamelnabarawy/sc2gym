import sys

import gym
import gflags as flags

# noinspection PyUnresolvedReferences
import sc2gym.envs

import time
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

    env = gym.make("SC2MoveToBeacon-v0")
    obs = env.reset()

    done = False
    while not done:
        action = move_to_beacon(obs)
        obs, reward, done, _ = env.step(action)

    env.save_replay(env.map_name)


def move_to_beacon(obs):
    if _MOVE_SCREEN in obs.observation["available_actions"]:
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        if not neutral_y.any():
            return actions.FunctionCall(_NO_OP, [])
        target = [int(neutral_x.mean()), int(neutral_y.mean())]
        return [_MOVE_SCREEN, _NOT_QUEUED, target]
    else:
        return [_SELECT_ARMY, _SELECT_ALL]


if __name__ == "__main__":
    main()
