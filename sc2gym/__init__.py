from gym.envs.registration import register

__author__ = 'Islam Elnabarawy'

register(
    id='SC2MoveToBeacon-v0',
    entry_point='sc2gym.envs:SC2MiniGameEnv',
    kwargs={
        'map_name': 'MoveToBeacon'
    }
)
