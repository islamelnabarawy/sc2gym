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
from sc2gym.envs.movement_minigame import BaseMovement1dEnv, BaseMovement2dEnv

__author__ = 'Islam Elnabarawy'

_MAP_NAME = 'MoveToBeacon'


class MoveToBeacon1dEnv(BaseMovement1dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)


class MoveToBeacon2dEnv(BaseMovement2dEnv):
    def __init__(self, **kwargs):
        super().__init__(map_name=_MAP_NAME, **kwargs)
