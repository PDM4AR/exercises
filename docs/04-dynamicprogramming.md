# Dynamic Programming :computer:

| _Prerequisites_:    | [Preliminaries](00-preliminaries.md) | [Hello-world](01-helloworld.md)|

In this programming exercise you will implement _Value_ and _Policy iterations_ for a particular stationary Markov
Decision Process (MDP).

You (autonomous robot) will be parachuted in a remote area of the planet for a rescue mission. You need to compute the optimal policy to reach the goal cell (visualized in RED). The world is modeled as a 2D grid, which is represented through a NxM matrix (numpy array). Rows and columns representing the “x” and “y” coordinates of the robot, respectively.

The area seems to be a tropical rainforest. Some cells in the map are simply ``GRASS`` (green), it will take you 1 time step to cross them. Some others are of type SWAMP (light blue), it will take you 5 time step to cross them. The reward at the GOAL cell is +10. For now, consider also the starting cell ``START`` as a ``GRASS`` cell.
When in a specific cell, the robot can move ``SOUTH, NORTH, EAST, WEST`` and if arrived at the ``GOAL``, it can ``STAY`` (actions and cells described in exercises/ex04/structures.py). 
The planet's atmosphere is very foggy and when the robot decides for an action, it may not end up where initially planned. In fact, for all other transitions, the following probabilities are given:
- When in ``GRASS``: all chosen transitions (``SOUTH, NORTH, EAST, WEST``) happen with probability of 0.75, the remaining 0.25 is split among the other 3 transitions not chosen.
- When in a ``SWAMP``: because it is also harder to move, with probability 0.25, the robot will not be able to move out of the current cell regardless of the chosen action. Any chosen transition will occur with 0.5 probability and because it is still foggy, the robot may end up with equal probability in one of the 3 remaining transitions.
- When in the ``GOAL`` the robot will ``STAY``  with probability of 1.
Moreover, although the robot cannot go directly out of the map, it may end up there accidentally, because this planet is flat the robot would fall off the planet and be transported back to the START cell.

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
    def __init__(self, grid: NDArray[np.int], gamma: float = .9):
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

#### Expected outcome

If your algorithm works, in the report you should find some results similar to this:

![image](https://user-images.githubusercontent.com/18750753/138459233-64bf90b9-526f-4d93-a919-5b9786ef4e2f.png)

On the left the Value function is visualized as a heatmap. On the right you can see the map with the original cells (
grass, swamps, goal,...) and the corresponding optimal policy.

### Test cases and performance criteria

The algorithms are going to be tested on different MDPs, each containing randomly located queries (start & goal cells).
You will be able to test your algorithms on some test cases with given solution, the outputted `Policy` and `ValueFunc` will be compared to the solution. 
After running the exercise, you'll find reports in `out/[exercise]/` for each test case. There you'll be able to visualize the MDPS, your output and the solution.
These test cases are not graded but serve as a guideline for how the exercise will be graded overall.

The final evaluation will combine 3 metrics lexicographically <number of solved cases, policy_accuracy, value_func_mspa, solve_time>:
* **policy_accuracy**: This metric will evaluate the accuracy of your `Policy`, in particular, it averages for each state of the MDP the number of correct actions (# of correct actions)/(# of states). Thus, policy_accuracy will be in the interval [0, 1].
* **value_func_mspa**: This metric will evaluate the accuracy of your `ValueFunc`. It is a measure of the mean accuracy of the given `ValueFunc` and it is calculated as: $ 1 - \frac{1}{n}\sum \frac{|VF^{gt} - VF|}{VF^{gt}}$.     
$VF^{gt}$ and $VF$ are the ground truth and your `ValueFunc` respectively. value_func_mspa will be in the interval $(-\infty, 1]$ where 1 means $VF^{gt} = VF$.
* **solve_time**: As your algorithms will be tested on graphs of increasing size, the efficiency of your code will be measured in terms of process time required. How do you expect the heuristic in A* to affect its solve time?

### Update your repo and run exercise

Make sure to update your repo before running the exercise.
Please refer to [Hello World](01-helloworld.md) for instructions.




