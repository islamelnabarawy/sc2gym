from gym.envs.registration import register

__author__ = 'Islam Elnabarawy'

register(
    id='SC2MoveToBeacon-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'MoveToBeacon'
    }
)

register(
    id='SC2CollectMineralShards-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'CollectMineralShards'
    }
)

register(
    id='SC2FindAndDefeatZerglings-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'FindAndDefeatZerglings'
    }
)

register(
    id='SC2DefeatRoaches-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'DefeatRoaches'
    }
)

register(
    id='SC2DefeatZerglingsAndBanelings-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'DefeatZerglingsAndBanelings'
    }
)

register(
    id='SC2CollectMineralsAndGas-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'CollectMineralsAndGas'
    }
)

register(
    id='SC2BuildMarines-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'BuildMarines'
    }
)
