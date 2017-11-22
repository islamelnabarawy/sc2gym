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
from gym.envs.registration import register

__author__ = 'Islam Elnabarawy'

register(
    id='SC2Game-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={}
)

register(
    id='SC2MoveToBeacon-v0',
    entry_point='sc2gym.envs:MoveToBeacon1dEnv',
    kwargs={}
)

register(
    id='SC2MoveToBeacon-v1',
    entry_point='sc2gym.envs:MoveToBeacon2dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v0',
    entry_point='sc2gym.envs:CollectMineralShards1dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v1',
    entry_point='sc2gym.envs:CollectMineralShards2dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v2',
    entry_point='sc2gym.envs:CollectMineralShardsGroupsEnv',
    kwargs={}
)

register(
    id='SC2FindAndDefeatZerglings-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={
        'map_name': 'FindAndDefeatZerglings'
    }
)

register(
    id='SC2DefeatRoaches-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={
        'map_name': 'DefeatRoaches'
    }
)

register(
    id='SC2DefeatZerglingsAndBanelings-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={
        'map_name': 'DefeatZerglingsAndBanelings'
    }
)

register(
    id='SC2CollectMineralsAndGas-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={
        'map_name': 'CollectMineralsAndGas'
    }
)

register(
    id='SC2BuildMarines-v0',
    entry_point='sc2gym.envs:SC2GameEnv',
    kwargs={
        'map_name': 'BuildMarines'
    }
)
