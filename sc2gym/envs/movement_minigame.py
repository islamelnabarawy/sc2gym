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
import numpy as np
from gym import spaces
from pysc2.lib import actions, features

from sc2gym.envs import SC2GameEnv

__author__ = 'Islam Elnabarawy'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_SCALE = features.SCREEN_FEATURES.player_relative.scale

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]


class BaseMovement1dEnv(SC2GameEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._action_space = None
        self._observation_space = None

    def reset(self):
        super().reset()
        return self._post_reset()

    def _post_reset(self):
        obs, reward, done, info = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._extract_observation(obs)
        return obs

    def step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._safe_step(action)
        if obs is None:
            return None, 0, True, {}
        obs = self._extract_observation(obs)
        return obs, reward, done, info

    @property
    def observation_space(self):
        if self._observation_space is None:
            self._observation_space = self._get_observation_space()
        return self._observation_space

    def _get_observation_space(self):
        screen_shape = (1, ) + self.observation_spec[0]["feature_screen"][1:]
        space = spaces.Box(low=0, high=_PLAYER_RELATIVE_SCALE, shape=screen_shape)
        return space

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = self._get_action_space()
        return self._action_space

    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.Discrete(screen_shape[0] * screen_shape[1] - 1)

    def _extract_observation(self, obs):
        obs = obs.observation["feature_screen"][_PLAYER_RELATIVE]
        obs = obs.reshape(self.observation_space.shape)
        return obs

    def _translate_action(self, action):
        if action < 0 or action > self.action_space.n:
            return [_NO_OP]
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        target = list(np.unravel_index(action, screen_shape))
        return [_MOVE_SCREEN, _NOT_QUEUED, target]


class BaseMovement2dEnv(BaseMovement1dEnv):
    def _get_action_space(self):
        screen_shape = self.observation_spec[0]["feature_screen"][1:]
        return spaces.MultiDiscrete([s-1 for s in screen_shape])

    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < 0 or act > self.action_space.nvec[ix]:
                return [_NO_OP]
        return [_MOVE_SCREEN, _NOT_QUEUED, action]
