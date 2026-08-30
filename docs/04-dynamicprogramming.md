# Dynamic Programming :computer:

| _Prerequisites_:    | [Preliminaries](00-preliminaries.md) | [Hello-world](01-helloworld.md)|

In this exercise you will implement _Value_ and _Policy iterations_ to solve Markov
Decision Processes (MDPs). The exercise has two parts. In **Part 1** you model a grid-world
problem as an MDP and solve it. In **Part 2** the world changes in three small ways, and each
change quietly breaks the Markov property of the grid: knowing the robot's cell is no longer
enough to predict its future. Your job, for each case, is to figure out what the state needs
to remember, turn the problem back into an MDP, and solve it with the very same Value and
Policy Iteration you wrote for Part 1.

You are an operator on a distant planet at a base responsible for deploying autonomous survey
robots looking for unobtainium. Your job is to send a surveying robot to a location specified
by your company's client.

For each contract, you can deploy several robots, but only one at a time - corporate requires
you to have at most one active robot in the field at any given time to try to save money.
You can always choose to abandon an active robot and deploy a new one from the base.
In some cases, an active robot might break down - then, you are forced to abandon it.
However, deploying the second and every next robot for a contract costs you money (the first
robot is covered by the client).

Once a robot reaches the goal location, it stays there and delivers data reports to the
client, thus fulfilling the contract. For this, you receive a reward from the client.
Your robot should reach the goal location as fast as possible since the client is entitled
to compensation for the time it takes you to fulfill the contract.

Your task is to create a plan for each given contract that maximizes your profit.
You choose to model this problem as a Markov Decision Process (MDP) and use dynamic
programming to compute offline the optimal policy for each mission.

## Part 1: the base problem

The world is modeled as a 2D grid, which is represented through a _MxN_ matrix (numpy array).
Rows and columns represent the $i$ and $j$ coordinates of the robot, respectively.
The area around you is a tropical rainforest, which can be modeled in the grid with the
following types of cells:
- ``GRASS`` (green) - it will take the robot 1 hour to cross this cell.
- ``SWAMP`` (light blue) - it will take the robot 2 hours to cross this cell.
- ``CLIFF`` (black) - untraversable cell. If the robot tries to move in this cell, it will
  break down and you will need to deploy a new robot from the base.
- ``GOAL`` (red) - the goal location you need to survey.
- ``START`` (yellow) - the location of your base, it can be considered a ``GRASS`` cell.
  The 8 cells close to the ``START`` cell are always ``GRASS`` or ``GOAL`` cells, except for
  the 4 in the corners that could also be ``SWAMP``, and there are at least 2 cells between
  the ``START`` cell and the edge of the map or a ``CLIFF`` cell. In other words, the robot
  will never break down in the ``START`` cell and the four neighboring cells.

The time required to cross each cell corresponds to the time a robot needs to leave this cell
(e.g. leaving the ``START`` cell takes 1 hour).
When in a specific cell, you can plan for the robot to take one of the following actions:
- move in a selected direction: ``SOUTH, NORTH, EAST, WEST``
- give up and ``ABANDON`` its mission (this implies the deployment of a new robot from the
  base),
- ``STAY`` if arrived at the ``GOAL``.

The goal of your policy is to maximize profit (use 1k USD as the default unit):
- You keep receiving a bonus of 50k USD for your robot surveying (staying at) the goal
  location.
- For each hour of the time it takes you to fulfill the contract, your client is entitled to
  a compensation of 1k USD. No compensation is paid if the robot is already at the goal
  location.
- The deployment of each new robot costs you 10k USD (the first robot is covered by the
  client).

The planet's atmosphere is very foggy and when the robot decides to move in a specific
direction, it may not end up where initially planned. Sometimes, when trying to move, it
might also break down and be forced to abandon its mission. In fact, for all transitions, the
following probabilities are given:
- In ``GRASS``:
  - If the robot chooses to move (``SOUTH, NORTH, EAST, WEST``), the chosen transition will
    happen with a probability of 0.75. The remaining 0.25 is split among the other 3 movement
    transitions. The robot will not break down in this cell.
  - If the robot chooses to give up, it will ``ABANDON`` its mission with probability 1.0.
- In ``SWAMP``:
  - Because it is harder to move, if the robot chooses to move
    (``SOUTH, NORTH, EAST, WEST``), the chosen transition will happen with a probability of
    0.5. With probability 0.25, the robot will move to one of the other neighboring cells
    (with all of them equally likely). With probability 0.2, the robot will not be able to
    move out of the cell (it will stay in the cell). With probability 0.05 the robot will
    break down and will need to ``ABANDON`` its mission.
  - If the robot chooses to give up, it will ``ABANDON`` its mission with probability 1.0.
- In ``CLIFF``:
  - The robot will break down with probability 1.0.
- When in the ``GOAL`` the robot will ``STAY`` with probability of 1.0.
- The robot cannot directly pick an action that would take it outside the map or to a
  ``CLIFF`` cell. However, it may be that the robot ends up out of the map or in a ``CLIFF``
  cell as described by the movement transition probabilities above. If this happens, the
  robot breaks down.

If the robot breaks down or chooses to ``ABANDON`` the mission, a new robot is deployed in
the ``START`` cell, which costs you 10k USD.

### Hints
- You can model the ``ABANDON`` action transition as a transition in your MDP from the
  current cell to the ``START`` cell with the cost of deploying a new robot.
- If the robot chooses a movement action in any cell, it will take it the time specified for
  this cell type to try to perform this action. An attempt to move out of the ``SWAMP`` cell
  always takes 2 hours, even if the robot ends up staying in the cell or breaking down.
- Note that the robot will never break down in the ``START`` cell or in its four neighboring
  cells because of the assumptions on their type and location on the map.

## Part 2: new states

Everything above still holds. In each of the three cases below, one short story change breaks
the Markov property of the bare grid; you decide what the state must remember to restore it.
The state becomes `(i, j, z)` where `z` is defined per case, and your value function and
policy become arrays of shape `(M, N, Z)`: one grid slice per value of `z`. Each case is
solved and graded separately.

### Case 1: Momentum ("the robot keeps rolling")

The robots' wheels carry momentum: slips lean toward wherever the robot moved last hour, and
turning around against your own motion is hard.

Let `h` be the direction the robot **actually moved** last hour, one of
`{-, NORTH, WEST, SOUTH, EAST}`, where `-` means it did not move (it stayed in place, or a
fresh robot was just deployed). When the robot chooses a movement action `u`:

| last heading h | intended u | slip directions | stay | break |
|---|---|---|---|---|
| `-` | as in Part 1 (grass 0.75, swamp 0.50) | uniform, as in Part 1 | Part 1 | Part 1 |
| h = u | +0.10 (grass 0.85, swamp 0.60) | uniform over the rest | Part 1 | Part 1 |
| h opposite of u | -0.10 (grass 0.65, swamp 0.40) | h: 0.25, others: 0.05 | Part 1 | Part 1 |
| h perpendicular to u | unchanged | h: 0.15, others: 0.05 | Part 1 | Part 1 |

After the hour: `h` becomes the direction the robot actually moved. If it stayed in place, or
broke down, or you chose `ABANDON` (a fresh robot is deployed at START), then `h = -`.

### Case 2: Forecast ("the base radios the fog")

Each hour the base transmits a forecast for the coming hour's fog. Forecasts are independent
across hours, with `P(FOGGY) = 0.3`. The forecast does not change the world; it only tells
you how this hour's move will behave:

| forecast f | GRASS intended / each slip | SWAMP intended / each slip | stay | break |
|---|---|---|---|---|
| CLEAR | 0.75 / 0.25 ÷ 3 (as in Part 1) | 0.50 / 0.25 ÷ 3 | 0.20 | 0.05 |
| FOGGY | 0.55 / 0.15 | 0.30 / 0.15 | 0.20 | 0.05 |

(The forecast's accuracy is already folded into these numbers.) Think carefully about whether
something that changes nothing physical still needs to be part of the state.

### Case 3: Glitch ("bad wheel days")

The robots' wheels sometimes glitch, and once the glitching starts it tends to last a while.
Let `g in {OK, GLITCHY}`:

- Movement: when `g = OK`, exactly as in Part 1. When `g = GLITCHY`, the intended-direction
  probability drops by 0.15 and the slip is uniform (grass 0.60 intended / 0.1333 each slip;
  swamp 0.35 intended / 0.1333).
- The glitch evolves on its own each hour: `P(OK -> GLITCHY) = 0.1` and
  `P(GLITCHY -> OK) = 0.3`.
- A freshly deployed robot has new wheels: after a breakdown or `ABANDON`, `g = OK`.
  (Yes, this has a consequence worth noticing.)

**Modeling statement (required, before the code).** For each case, a few sentences: what your
state is, why the grid alone stops being Markov, and why your state restores it. This is
graded by a short rubric; it is where wrong state choices get caught and discussed instead of
silently punished.

## Tasks

### Data structure

Actions, states, Value function and Policy are defined as follows
(exercises/ex04/structures.py):

```python
@unique
class Action(IntEnum):
    NORTH = 0
    WEST = 1
    SOUTH = 2
    EAST = 3
    STAY = 4
    ABANDON = 5


State = tuple[int, int]
"""The Part-1 state is a tuple of two ints."""


@unique
class Cell(IntEnum):
    GOAL = 0
    START = 1
    GRASS = 2
    SWAMP = 3
    CLIFF = 5


AugmentedState = tuple[int, int, int]
"""The Part-2 state is (i, j, z); z orderings are fixed by these enums:"""


@unique
class Heading(IntEnum):
    NONE = 0; NORTH = 1; WEST = 2; SOUTH = 3; EAST = 4

@unique
class Fog(IntEnum):
    CLEAR = 0; FOGGY = 1

@unique
class Gear(IntEnum):
    OK = 0; GLITCHY = 1


ValueFunc = NDArray[np.float64]   # (M, N) in Part 1, (M, N, Z) in Part 2
Policy = NDArray[np.int64]        # (M, N) in Part 1, (M, N, Z) in Part 2

OptimalActions = NDArray[np.object_]
"""
Type Alias for the all optimal actions per state. It is a numpy array of list objects where
each list contains the optimal actions that are equally good for a given state. It is the
type of the ground truth policy that your solution will be compared against. You are not
required to use this type in your solution.
"""
```

### The MDP models

The first subtask is to implement the missing methods in `exercises/ex04/mdp.py`.
These methods will be useful when implementing value and policy iteration.

For each class you need to fill in `get_transition_prob`, which returns the probability of
transitioning from a state to another given an action, and `stage_reward`, which returns the
reward for that transition. The signatures are identical across all four classes; only the
`State` widens to `AugmentedState` in Part 2:

```python
class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo


class MomentumGridMdp(AugmentedGridMdp): ...   # Z = 5
class FogGridMdp(AugmentedGridMdp): ...        # Z = 2
class GlitchGridMdp(AugmentedGridMdp): ...     # Z = 2
```

Feel free to add more methods in case you need to.
You must not change the names and signatures of `get_transition_prob` and `stage_reward`.

### Value Iteration and Policy Iteration

Implement `solve` in ``exercises/ex04/value_iteration.py`` and
``exercises/ex04/policy_iteration.py``:

```python
def solve(grid_mdp, max_iters: int | None = None) -> tuple[ValueFunc, Policy]:
    ...
```

One solver serves both parts: if you write it to iterate over "the states of the MDP" it runs
on Part 2 unchanged; only the model underneath grows. If `max_iters = k` is given, return the
state of your algorithm after exactly k iterations (k synchronous sweeps for VI; k
evaluate-improve cycles for PI), starting from V = 0 and, for PI, from the first admissible
action per state, breaking ties by action order.

> **Note**: The optimal value function is unique, but the optimal policy is not. You can
> return any optimal policy; the ground truth stores all optimal actions per state, so your
> tie-breaking never costs you.

### Help for modeling the MDP

Correctly modeling the MDPs is crucial. For Part 1 we provide the admissible action set and
ground truth transition probabilities and rewards for selected cells of the 5x5 example map;
for Part 2, `get_transition_prob` is additionally evaluated on sampled augmented triples.
The coordinates of the cells are given in the format `(row, column)` starting from the top
left corner of the grid. All axes are 0-indexed.

#### Admissible action set (Part 1, 5x5 example)

```<current state>: [list of admissible actions]```

```
- (0, 1): [SOUTH, EAST, ABANDON]
- (1, 2): [NORTH, WEST, SOUTH, EAST, ABANDON]
- (2, 0): [NORTH, SOUTH, EAST, ABANDON]
- (4, 3): [NORTH, WEST, ABANDON]
```

#### Ground truth transition probabilities and rewards (Part 1, 5x5 example)

```(<current state>, <action>): (<next state>, <probability>, <reward>)```

```
- ((0, 1), SOUTH):   ((1, 1), 0.75, -1), ((0, 2), 0.0833, -1), ((2, 2), 0.1667, -11)
- ((0, 1), ABANDON): ((2, 2), 1.0, -10)
- ((1, 2), WEST):    ((1, 1), 0.75, -1), ((0, 2), 0.0833, -1), ((1, 3), 0.0833, -1), ((2, 2), 0.0833, -1)
- ((2, 0), SOUTH):   ((3, 0), 0.75, -1), ((1, 0), 0.0833, -1), ((2, 1), 0.0833, -1), ((2, 2), 0.0833, -11)
- ((3, 3), WEST):    ((3, 2), 0.5, -2), ((2, 3), 0.0833, -2), ((3, 4), 0.0833, -2), ((4, 3), 0.0833, -2), ((3, 3), 0.2, -2), ((2, 2), 0.05, -12)
```

### Practice maps (not graded)

Besides the graded test maps, we publish 12 extra practice maps of varied sizes with full
solutions for Part 1 and every Part-2 case, so you can check your implementation yourself:

```python
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises_def.ex04.practice import check_solution

check_solution(ValueIteration)                       # everything
check_solution(ValueIteration, cases=["base"])       # Part 1 only
check_solution(ValueIteration, cases=["forecast"])   # one Part-2 case
```

It prints, per map, the same policy accuracy and value R2 the graded evaluation computes. If
a map scores below 1.0, that map and case is where to look.


### Make your own maps, and verify without an answer key

You can generate as many extra maps as you like and check your solutions on
them with no ground truth at all, because an optimal DP solution carries
checkable certificates:

```python
from pdm4ar.exercises_def.ex04.selfcheck import random_map, self_check

grid = random_map((10, 10), seed=7)      # goal always reachable
mdp = FogGridMdp(grid)
v_vi, p_vi = ValueIteration.solve(mdp)
v_pi, p_pi = PolicyIteration.solve(mdp)
self_check(mdp, "forecast", v_vi, p_vi, v_pi)
```

`self_check` runs, using only your own model and solutions:
- probabilities sum to 1 for every (state, action), and `ABANDON` sends all
  mass to `START`;
- V <= 500 everywhere and V(goal) = 500 exactly (the geometric series of the
  +50 bonus);
- forecast: V(., CLEAR) >= V(., FOGGY); glitch: V(., OK) >= V(., GLITCHY) at
  every cell;
- your Value Iteration and Policy Iteration agree with each other;
- a Monte-Carlo rollout of your policy under your own model reproduces
  V(start) within statistical error.

**The one exact certificate is yours to implement.** V is optimal if and only
if one more Bellman backup changes nothing:

    max over states |V(s) - max over a of sum over s' P(s'|s,a) [r + gamma V(s')]| ~ 0

and your policy is greedy with respect to V. Apply one sweep of your own
backup to your converged V; if anything moves by more than your convergence
tolerance, you are not done. We deliberately do not ship this check as code:
writing it is five lines you already wrote inside Value Iteration, and it is
the honest answer to "how do I know I am finished" for any DP problem.

Two caveats to keep in mind: all of these checks verify your solution against
YOUR transition model, so a systematically wrong model can pass them; the
published worked examples above and the practice maps (with real answers) are
your anchors for the model itself. And passing every check is necessary, not
sufficient; the Bellman certificate is the one that closes the argument.

### Test cases and performance criteria

The algorithms are tested on the maps of both parts. After running the exercise, you will
find the reports in `out/04/` for each test case; Part-2 reports show one heatmap per z
slice, so you can literally see how the optimal route changes between a CLEAR and a FOGGY
forecast, or how momentum makes the robot prefer not to turn around.

The final evaluation combines, exactly as before: ratio of completed cases,
transition_prob_accuracy, average policy_accuracy, average value_func_R2, and average
solve_time. Policy accuracy counts an action as correct if it is any optimal action of that
state. The value function and policy of `CLIFF` cells are excluded from the evaluation.

Note: both algorithms must be your own. The grader compares your Value and Policy Iteration
submissions against each other, and requesting intermediate iterates (`max_iters`) makes the
two algorithms distinguishable even though they agree at convergence.
