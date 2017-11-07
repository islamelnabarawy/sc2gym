from examples.base_example import BaseExample

__author__ = 'Islam Elnabarawy'


_ENV_NAME = "SC2CollectMineralShards-v2"
_VISUALIZE = False
_STEP_MUL = None
_NUM_EPISODES = 1


class MinigameRandom(BaseExample):
    def __init__(self, visualize=False, step_mul=None) -> None:
        super().__init__(_ENV_NAME, visualize, step_mul)

    def get_action(self, env, obs):
        return env.action_space.sample()


def main():
    example = MinigameRandom(_VISUALIZE, _STEP_MUL)
    rewards = example.run(_NUM_EPISODES)
    print('Total reward: {}'.format(rewards.sum()))
    print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    print('Minimum reward: {}'.format(rewards.min()))
    print('Maximum reward: {}'.format(rewards.max()))


if __name__ == "__main__":
    main()
