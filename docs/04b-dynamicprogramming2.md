# Dynamic Programming II: New States (Exercise 04b)

| _Prerequisites_: | [Dynamic Programming (Exercise 04)](04-dynamicprogramming.md) |

In Exercise 04 you modeled the survey-robot problem as an MDP and solved it
with Value and Policy Iteration. In this exercise the world changes in three
small ways, and each change quietly breaks the Markov property of the grid:
knowing the robot's cell is no longer enough to predict its future. Your job,
for each case, is to figure out what the state needs to remember, turn the
problem back into an MDP, and solve it with the Value and Policy Iteration you
already wrote.

You are still the operator on the distant planet. Movement, swamps, cliffs,
breakdowns, and all rewards behave **exactly as in Exercise 04** unless a case
says otherwise. Two differences apply everywhere:

- The maps in this exercise contain **no WONDERLAND cells**.
- The state is no longer just a cell. For each case the state is
  `(i, j, z)` where `z` is defined below, and your value function and policy
  are arrays of shape `(M, N, Z)`: one grid slice per value of `z`.

Each case is a separate task, graded separately.

## Warm-up (not graded)

To see the method once, suppose a contract required picking up a package at a
cell P before the goal counts. The robot's cell alone is not Markov anymore:
two robots in the same cell, one carrying the package and one not, have
different futures. The fix is to remember the task progress,
`q in {not_yet, carrying}`, and plan on the product state `(i, j, q)`: the
value function becomes two stacked grids, the transition from `q = not_yet` to
`q = carrying` happens exactly when the robot enters P, and your Exercise 04
solver runs on the bigger state set without any change to the algorithm. Every
case below follows this pattern with a different `z`.

## Case 1: Momentum ("the robot keeps rolling")

The robots' wheels carry momentum: slips lean toward wherever the robot moved
last hour, and turning around against your own motion is hard.

Let `h` be the direction the robot **actually moved** last hour, one of
`{-, NORTH, WEST, SOUTH, EAST}`, where `-` means it did not move (it stayed in
place, or a fresh robot was just deployed). When the robot chooses a movement
action `u`:

| last heading h | intended u | slip directions | stay | break |
|---|---|---|---|---|
| `-` | as in ex04 (grass 0.75, swamp 0.50) | uniform, as in ex04 | ex04 | ex04 |
| h = u | +0.10 (grass 0.85, swamp 0.60) | uniform over the rest | ex04 | ex04 |
| h opposite of u | -0.10 (grass 0.65, swamp 0.40) | h: 0.25, others: 0.05 | ex04 | ex04 |
| h perpendicular to u | unchanged | h: 0.15, others: 0.05 | ex04 | ex04 |

After the hour: `h` becomes the direction the robot actually moved. If it
stayed in place, or broke down, or you chose `ABANDON` (a fresh robot is
deployed at START), then `h = -`.

## Case 2: Forecast ("the base radios the fog")

Each hour the base transmits a forecast for the coming hour's fog. Forecasts
are independent across hours, with `P(FOGGY) = 0.3`. The forecast does not
change the world; it only tells you how this hour's move will behave:

| forecast f | GRASS intended / each slip | SWAMP intended / each slip | stay | break |
|---|---|---|---|---|
| CLEAR | 0.75 / 0.25 ÷ 3 (as in ex04) | 0.50 / 0.25 ÷ 3 | 0.20 | 0.05 |
| FOGGY | 0.55 / 0.15 | 0.30 / 0.15 | 0.20 | 0.05 |

(The forecast's accuracy is already folded into these numbers.) Think
carefully about whether something that changes nothing physical still needs to
be part of the state.

## Case 3: Glitch ("bad wheel days")

The robots' wheels sometimes glitch, and once the glitching starts it tends to
last a while. Let `g in {OK, GLITCHY}`:

- Movement: when `g = OK`, exactly as in ex04. When `g = GLITCHY`, the
  intended-direction probability drops by 0.15 and the slip is uniform
  (grass 0.60 intended / 0.1333 each slip; swamp 0.35 intended / 0.1333).
- The glitch evolves on its own each hour: `P(OK -> GLITCHY) = 0.1` and
  `P(GLITCHY -> OK) = 0.3`.
- A freshly deployed robot has new wheels: after a breakdown or `ABANDON`,
  `g = OK`. (Yes, this has a consequence worth noticing.)

## Data structures

```python
from enum import IntEnum, unique

@unique
class Heading(IntEnum):
    NONE = 0; NORTH = 1; WEST = 2; SOUTH = 3; EAST = 4

@unique
class Fog(IntEnum):
    CLEAR = 0; FOGGY = 1

@unique
class Gear(IntEnum):
    OK = 0; GLITCHY = 1

State = tuple[int, int, int]        # (i, j, z)
ValueFunc = NDArray[np.float64]     # shape (M, N, Z)
Policy = NDArray[np.int64]          # shape (M, N, Z)
```

`Action` and `Cell` are the ones you know from Exercise 04. The z orderings
above are fixed; your output arrays must use them.

## Tasks

For each case, implement the two methods of the corresponding MDP class
(shown here for Case 2; `MomentumGridMdp` and `GlitchGridMdp` are identical in
shape):

```python
class FogGridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9): ...

    def get_transition_prob(self, state: State, action: Action,
                            next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo

    def stage_reward(self, state: State, action: Action,
                     next_state: State) -> float:
        # todo
```

Then solve each case with your Value Iteration and Policy Iteration from
Exercise 04. The solver interface is unchanged except for one optional
parameter used by the evaluation:

```python
def solve(grid_mdp, max_iters: int | None = None) -> tuple[ValueFunc, Policy]:
    ...
```

If `max_iters = k` is given, return the state of your algorithm after exactly
k iterations (k synchronous sweeps for VI; k evaluate-improve cycles for PI),
starting from V = 0 and, for PI, from the first admissible action per state,
breaking ties by action order.

**Modeling statement (required, before the code).** For each case, a few
sentences: what your state is, why the grid alone stops being Markov, and why
your state restores it. This is graded by a short rubric; it is where wrong
state choices get caught and discussed instead of silently punished.

## Expected outcome and evaluation

For each case you return the optimal `ValueFunc` and one optimal `Policy` as
`(M, N, Z)` arrays. Evaluation mirrors Exercise 04: `get_transition_prob` is
checked on sampled `(state, action, next_state)` triples, and your value
function and policy are compared per slice against the ground truth
(policy accuracy, value R2, solve time). The report shows one heatmap per
slice, so you can literally see, for example, how the optimal route changes
between a CLEAR and a FOGGY forecast, or how the momentum makes the robot
prefer not to turn around.

Note: both algorithms must be your own. The grader compares your Value and
Policy Iteration submissions against each other, and requesting intermediate
iterates (`max_iters`) makes the two algorithms distinguishable even though
they agree at convergence.

## Practice maps (not graded)

Besides the graded test maps, we publish 12 extra practice maps of varied
sizes with full solutions for every case, so you can check your
implementation yourself:

```python
from pdm4ar.exercises.ex04b.value_iteration import ValueIteration
from pdm4ar.exercises_def.ex04b.practice import check_solution

check_solution(ValueIteration)                      # all cases, all maps
check_solution(ValueIteration, cases=["forecast"])  # one case only
```

It prints, per map, the same policy accuracy and value R2 the graded
evaluation computes (policy accuracy counts ALL optimal actions as correct,
so your tie-breaking never costs you). If a map scores below 1.0, that map
and case is where to look.

## Hints

- Start by asking, for each case: standing in cell (i, j), what else must I
  know to predict the next hour? That answer is your `z`.
- The z of your **current** state conditions the current hour's move; the next
  z is part of the transition.
- In two of the three cases the transition kernel factorizes into a movement
  part and a z part. Finding that factorization keeps your
  `get_transition_prob` short and your solver fast; nothing forces you to
  enumerate a joint table.
- Your Exercise 04 solver needs no algorithmic change. If you wrote it to
  iterate over "all states of the MDP", it runs as is; if you hard-coded the
  `(M, N)` shape, generalize it once and it works for all three cases.
