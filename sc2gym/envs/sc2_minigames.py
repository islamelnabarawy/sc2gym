import logging

import gym
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id


class SC2MiniGameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}

    def __init__(self, map_name=None, step_mul=8, visualize=True) -> None:
        super().__init__()
        self._map_name = map_name
        self._visualize = visualize
        self._step_mul = step_mul
        self._env = None

        self.episode = 0
        self.num_step = 0

    def _step(self, action):
        self.num_step += 1
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
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _reset(self):
        if self._env is None:
            self._init_env()
        self.episode += 1
        self.num_step = 0
        logger.info("Episode %d starting...", self.episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        self._env = sc2_env.SC2Env(
            map_name=self._map_name,
            step_mul=self._step_mul,
            visualize=self._visualize
        )

    def close(self):
        self._env.close()
        super().close()

    @property
    def visualize(self):
        return self._visualize

    @visualize.setter
    def visualize(self, value):
        if self._env is not None:
            logger.warning("Setting visualize attribute after the game started has no effect.")
        else:
            self._visualize = value

    @property
    def map_name(self):
        return self._map_name

    @map_name.setter
    def map_name(self, value):
        if self._env is not None:
            logger.warning("Setting map_name attribute after the game started has no effect.")
        else:
            self._map_name = value

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
