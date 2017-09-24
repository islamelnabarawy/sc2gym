## PySC2 OpenAI Gym Environments

OpenAI Gym Environments for the StarCraft II PySC2 environment.

This is still a work in progress.

### Notes:
* The environment is currently defined the same way for all the mini games.
* The action_space and observation_space member variables are not being set,
instead there is an action_spec and observation_spec, which come directly
from the pysc2 environment.
* Per the Gym environment specifications, the reset function returns an
observation, and the step function returns a tuple
(observation, reward, done, info), where info is an empty dictionary and
the observation is the full observation object from the pysc2 environment.
The reward is the same as observation.reward, and done is equal true if
observation.step_type is LAST.
* In addition to step() and reset(), the environment defines a save_replay()
method, that takes a single parameter, replay_dir, which is the name of the
replay directory to save to inside the `StarCraft II/Replays/` folder.
* The examples folder contains an example of using this environment.
There are two functions defined in there, random_action() and move_to_beacon().
The random_action() function was tested with each of the environment. The
move_to_beacon() function has the scripted agent example from pysc2.
