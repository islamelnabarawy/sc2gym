from gym import spaces
from pysc2.lib import actions
from sc2gym.envs import SC2GameEnv
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'CollectMineralShards'

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_GROUP_RECALL = [0]
_GROUP_SET = [1]

_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_SINGLE = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_NO_OP = actions.FUNCTIONS.no_op.id


class CollectMineralShards1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)


class CollectMineralShards2dEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)


class CollectMineralShardsGroupsEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, step_mul=2, **kwargs)

    def _post_reset(self):
        obs = self._init_control_groups()
        obs = self._extract_observation(obs[0])
        return obs

    def _init_control_groups(self):
        obs = self._safe_step([_SELECT_ARMY, _SELECT_ALL])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [0]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [2]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [1]])
        obs = self._safe_step([_SELECT_UNIT, _SELECT_SINGLE, [1]])
        obs = self._safe_step([_CONTROL_GROUP, _GROUP_SET, [3]])
        return obs

    def _step(self, action):
        if _MOVE_SCREEN not in self.available_actions:
            self._init_control_groups()
        return super()._step(action)

    def _get_action_space(self):
        screen_shape = self.observation_spec["screen"][1:]
        return spaces.MultiDiscrete([(0, 3)] + [(0, s) for s in screen_shape])

    def _translate_action(self, action):
        for ix, act in enumerate(action):
            if act < self.action_space.low[ix] or act >= self.action_space.high[ix]:
                return [_NO_OP]
        self._safe_step([_CONTROL_GROUP, _GROUP_RECALL, [action[0] + 1]])
        return [_MOVE_SCREEN, _NOT_QUEUED, action[1:]]
