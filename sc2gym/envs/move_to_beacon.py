from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'MoveToBeacon'


class MoveToBeacon1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)


class MoveToBeacon2dEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)
