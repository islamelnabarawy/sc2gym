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
import logging

import gym
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id


class SC2GameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = {'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=64,
    )}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def step(self, action):
        return self._safe_step(action)

    def _safe_step(self, action):
        self._num_step += 1
        if action[0] not in self.available_actions:
            logger.warning("Attempted unavailable action: %s", action)
            action = [_NO_OP]
        try:
            obs = self._env.step([actions.FunctionCall(action[0], action[1:])])[0]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}
        except Exception:
            logger.exception("An unexpected error occurred while applying action to environment.")
            return None, 0, True, {}
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def reset(self):
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward so far, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        logger.info("Episode %d starting...", self._episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        args = {**self.default_settings, **self._kwargs}
        logger.debug("Initializing SC2Env with settings: %s", args)
        self._env = sc2_env.SC2Env(**args)

    def close(self):
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def settings(self):
        return self._kwargs

    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward
