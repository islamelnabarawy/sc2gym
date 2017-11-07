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

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self.episode = 0
        self.num_step = 0
        self.episode_reward = 0
        self.total_reward = 0

    def _step(self, action):
        return self._safe_step(action)

    def _safe_step(self, action):
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
        self.episode_reward += reward
        self.total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _reset(self):
        if self._env is None:
            self._init_env()
        if self.episode > 0:
            self.report_reward()
        self.episode += 1
        self.num_step = 0
        self.episode_reward = 0
        logger.info("Episode %d starting...", self.episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs

    def report_reward(self):
        logger.info("Episode %d ended with reward %d after %d steps.",
                    self.episode, self.episode_reward, self.num_step)
        logger.info("Got %d total reward so far, with an average reward of %g per episode",
                    self.total_reward, float(self.total_reward) / self.episode)

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        self._env = sc2_env.SC2Env(**self._kwargs)

    def _close(self):
        if self.episode > 0:
            self.report_reward()
        self._env.close()
        super()._close()

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
