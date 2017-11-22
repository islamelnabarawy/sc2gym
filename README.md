# PySC2 OpenAI Gym Environments

OpenAI Gym Environments for the StarCraft II PySC2 environment.

## Installation:

After cloning the repository, you can use the environments in
one of two ways:

1. Add the directory where you cloned the repo to your `PYTHON_PATH`
2. Install the package in development mode using pip: `pip install -e .`

If you use the first option, you need to manually make sure the
dependencies are installed.

The second option will install the package into your `pip` environment
as a link to the directory, so it will reflect the changes when
you `git pull` or make any changes to the code.

## Usage:

You need the following minimum code to run any environment:

Import gym and this package:

    import gym
    import sc2gym.envs

Import and initialize absl.flags: (this is due to `pysc2` dependency)

    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

Create and initialize the specific environment as indicated in the
next section.

## Available environemnts:

### SC2Game:

The full StarCraft II game environment. Initialize as follows:

    env = gym.make('SC2Game-v0')
    env.settings['map_name'] = '<desired map name>'

Versions:
- `SC2Game-v0`: The full game with complete access to action and
observation space.

#### Notes:
- The action space for this environment doesn't require the call to
`functionCall` like `pysc2` does. You just need to call it with an
array of action and arguments. For example:

        _SELECT_ARMY = actions.FUNCTIONS.select_army.id
        _SELECT_ALL = [0]
        env.step([_SELECT_ARMY, _SELECT_ALL])

    It will check the first element in the array (the action) and make
    sure it's available before trying to pass it along to the
    `pysc2` environment.
- This environment doesn't specify the `observation_space` and
`action_space` members like traditional `gym` environments. Instead,
it provides access to the `observation_spec` and `action_spec` objects
from the `pysc2` environment.

### MoveToBeacon:

The MoveToBeacon mini game. Initialize as follows:

    env = gym.make('SC2MoveToBeacon-v0')

Versions:
- 'SC2MoveToBeacon-v0': The observation is a `[1, 64, 64]` numpy
array that represents the `obs.observation['screen'][_PLAYER_RELATIVE]`
plane from the `pysc2` observation. The action is a number
between 0 and 4095 (64x64-1), which is a 1-d representation of the
screen coordinates to move the marine towards. The environment
pre-selects the marine at the start of each episode.
- 'SC2MoveToBeacon-v1': The observation is a `[1, 64, 64]` numpy
array that represents the `obs.observation['screen'][_PLAYER_RELATIVE]`
plane from the `pysc2` observation. The action is an array of
two numbers, each between 0 and 63, representing the screen coordinates
to move the marine towards. The environment pre-selects the marine at
the start of each episode.

### CollectMineralShards:

The CollectMineralShards mini game. Initialize as follows:

    env = gym.make('SC2CollectMineralShards-v0')

Versions:
- 'SC2CollectMineralShards-v0': The observation is a `[1, 64, 64]` numpy
array that represents the `obs.observation['screen'][_PLAYER_RELATIVE]`
plane from the `pysc2` observation. The action is a number
between 0 and 4095 (64x64-1), which is a 1-d representation of the
screen coordinates to move the marines towards. The environment
pre-selects both marines at the start of each episode.
- 'SC2CollectMineralShards-v1': The observation is a `[1, 64, 64]` numpy
array that represents the `obs.observation['screen'][_PLAYER_RELATIVE]`
plane from the `pysc2` observation. The action is an array of
two numbers, each between 0 and 63, representing the screen coordinates
to move the marines towards. The environment pre-selects the marine at
the start of each episode.
- 'SC2CollectMineralShards-v2': The observation is a `[2, 64, 64]` numpy
array that represents the `obs.observation['screen'][_PLAYER_RELATIVE]`
and the `obs.observation['screen'][_SELECTED]` planes from the `pysc2`
observation. The action is an array of three numbers. The first number
is between 0 and 2, representing which control group to move.
The remaining two numbers are integers between 0 and 63, representing the screen coordinates
to move the marines in this control group towards.
The environment automatically creates the following control groups at
the start of each episode.
__Each episode starts with control group 3 pre-selected.__
    * Group 1 [index 0]: Both marines
    * Group 2 [index 1]: Marine 1
    * Group 3 [index 2]: Marine 2


### General Notes:
* Per the Gym environment specifications, the reset function returns an
observation, and the step function returns a tuple
(observation, reward, done, info), where info is an empty dictionary and
the observation is the observation object from the pysc2 environment.
The reward is the same as observation.reward, and done is equal true if
observation.step_type is LAST.
* In addition to `step()` and `reset()`, the environments define a
`save_replay()` method, that takes a single parameter, `replay_dir`,
which is the name of the replay directory to save to inside the
`StarCraft II/Replays/` folder.
* All the environments have `action_spec` and `observation_spec` properties,
in addition to the `action_space` and `observation_space` properties defined
for the mini game environments.
* All the environments have the following additional properties:
    - `episode`: The current episode number
    - `num_step`: The total number of steps taken
    - `episode_reward`: The total reward received this episode
    - `total_reward`: The total reward received for all episodes
* The examples folder contains examples of using the various environments.

---
    
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