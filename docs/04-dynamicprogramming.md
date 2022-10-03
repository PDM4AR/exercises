# Dynamic Programming :computer:

| _Prerequisites_:    | [Preliminaries](00-preliminaries.md) | [Hello-world](01-helloworld.md)|

In this programming exercise you will implement _Value_ and _Policy iterations_ for a particular stationary Markov
Decision Process (MDP).

You (autonomous robot) you will be parachuted in a remote area of the planet for a rescue mission. You need to compute
the optimal policy to reach the goal cell (visualized in **RED**). The world is modeled as a 2D grid, which is
represented through a _NxM_ matrix (numpy array). Rows and columns representing the "x" and "y" coordinates of the
robot, respectively.

The area seems to be a tropical rainforest. Some cells in the map are simply ``GRASS`` (green), it will take you 1 time
step to cross them. Some others are of type ``SWAMP`` (light blue), it will take you 5 time step to cross them. Hence
you can set the respective reward to be -1 for ``GRASS`` and -10 for ``SWAMP``. The reward at the ``GOAL`` cell is +10.
For now, consider also the starting cell ``START`` as a ``GRASS`` cell.

When in a specific cell, the robot can move ``SOUTH, NORTH, EAST, WEST`` (if not next to a boundary) and if arrived at
the ``GOAL``, it can ``STAY`` (actions and cells described in ``exercises/ex04/structures.py``). Applying an action from
any given cell to get to an adjacent cell is successful with a probability of 1.

## Tasks

### Data structure

Actions, states, Value function and Policy are defined as follows:

```python
from enum import IntEnum, unique
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@unique
class Action(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3
    STAY = 4


State = Tuple[int, int]
"""The state on a grid is simply a tuple of two ints"""


@unique
class Cell(IntEnum):
    GOAL = 0
    START = 1
    GRASS = 2
    SWAMP = 3


Policy = NDArray[np.int]
"""Type Alias for the policy, the integer should correspond to one of the Actions"""
ValueFunc = NDArray[np.float]
"""Type Alias for the value function"""
```

The first subtask consists in implementing the missing methods in `exercises/ex04/mdp.py`. These methods will be useful
when implementing value and policy iteration.

```python
class GridMdp:
    def __init__(self, grid: NDArray[np.int], gamma: float = .7):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo

    def stage_reward(self, state: State, action: Action) -> float:
# todo


```

Feel free to add more methods in case you need. The method ``get_transition_prob`` returns the probability of
transitioning from a state to another given an action. Finally, the method ``stage_reward``returns the reward
corresponding to applying an action at the current state.

#### Value Iteration

With start with value iteration. You need to implement the following methods in ``exercises/ex04/value_iteration.py``:

```python
class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here

        return value_func, policy
```

#### Policy iteration

For policy iteration, you need to implement the following methods in ``exercises/ex04/policy_iteration.py``:

```python
class PolicyIteration(GridMdpSolver):

    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> Tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here

        return value_func, policy
```

### Update your repo

Update your repo using

```
make update
```

this will put the new exercises in your forked repo. If you get some merge conflicts it is because you might have
modified/moved files that you were not supposed to touch (i.e., outside the ``exercises`` folder).

### Run the exercise

Now you can run the exercise.

```
make run-exercise4
```

The report will provide visual information regarding the computed values/policies.

#### Expected outcome

If your algorithm works, in the report you should find some results similar to this:

![image](https://user-images.githubusercontent.com/18750753/138459233-64bf90b9-526f-4d93-a919-5b9786ef4e2f.png)

On the left the Value function is visualized as a heatmap. On the right you can see the map with the original cells (
grass, swamps, goal,...) and the corresponding optimal policy.

#### Food for thoughts

* We specified the reward of the starting cell as a ``GRASS`` cell (-1). What if you now change it to a value >=0?
* The transition have been considered deterministic so far.
Would anything change if now when you apply an action with probability $.75$ you move by 1 cell but with p(.25) you move by two in the desired direction?
How would the pattern of the arrows change?
* How do you expect your solution to change if you change the discount factor? you can change it in `exercises_def/ex04/data.py`
