import logging

import gym
# from gym import spaces
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_NO_OP = actions.FUNCTIONS.no_op.id


class SC2MiniGameEnv(gym.Env):
    metadata = {'render.modes': [None, 'human']}

    step_mul = 8

    def __init__(self, map_name, visualize=True) -> None:
        super().__init__()
        self.map_name = map_name
        self._env = sc2_env.SC2Env(
            map_name,
            step_mul=self.step_mul,
            visualize=visualize
        )
        self.action_spec = self._env.action_spec()
        self.observation_spec = self._env.observation_spec()

        # self.action_space = spaces.Discrete(len(self.action_spec.functions))
        # self.observation_space = spaces.Box(low=0, high=100, shape=self.observation_spec['screen'])

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
        self.episode += 1
        self.num_step = 0
        logger.info("Episode %d starting...", self.episode)
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return obs

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)
