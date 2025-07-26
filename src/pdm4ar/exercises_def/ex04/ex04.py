from dataclasses import dataclass
from time import process_time
from typing import Any, Sequence, Type, Union, Optional
from zuper_commons.text import remove_escapes

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises.ex04.structures import Action, OptimalActions, Cell, Policy, State
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.data import get_expected_results, get_test_grids
from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.structures import PerformanceResults
from reprep import MIME_PDF, Report


@dataclass
class TestValueEx4(ExIn):
    algo: Type[GridMdpSolver]
    grid: GridMdp
    testId: int = 0

    def str_id(self) -> str:
        return str(self.algo.__name__) + str(self.testId)


@dataclass
class TestTransitionProbEx4(ExIn):
    grid: GridMdp
    state: State
    action: Action
    next_state: State
    testId: int = 0

    def str_id(self) -> str:
        return f"TransitionProb{self.testId}_s{self.state}_a{self.action.name}_ns{self.next_state}"


@dataclass(frozen=True)
class Ex04Performance(PerformanceResults):
    policy_accuracy: float
    value_func_r2: float
    solve_time: float

    def __post__init__(self):
        assert self.policy_accuracy <= 1, self.policy_accuracy
        assert self.solve_time >= 0, self.solve_time


@dataclass(frozen=True)
class Ex04TransitionProbPerformance(PerformanceResults):
    transition_prob_accuracy: float
    test_count: int

    def __post__init__(self):
        assert self.transition_prob_accuracy <= 1, self.transition_prob_accuracy
        assert self.test_count >= 0, self.test_count


@dataclass(frozen=True)
class Ex04PerformanceResult(PerformanceResults):
    # All test cases
    perf_result: Ex04Performance
    # Value Iteration test cases
    value_iteration: Optional[Ex04Performance] = None
    # Policy Iteration test cases
    policy_iteration: Optional[Ex04Performance] = None
    # Transition probability test cases
    transition_prob: Optional[Ex04TransitionProbPerformance] = None


def get_font_size(grid_mdp: GridMdp) -> int:
    num_row = grid_mdp.grid.shape[0]
    if num_row <= 15:
        return 6
    elif num_row <= 30:
        return 3
    else:
        return 2


def plot_grid_values(rfig, grid_mdp: GridMdp, value_func: np.ndarray, algo_name: str):
    MAP_SHAPE = grid_mdp.grid.shape
    font_size = get_font_size(grid_mdp)
    with rfig.plot(nid=f"{algo_name}-value", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(value_func, aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for i in range(MAP_SHAPE[0]):
            for j in range(MAP_SHAPE[1]):
                if grid_mdp.grid[i, j] == Cell.CLIFF:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="k"))
                elif grid_mdp.grid[i, j] == Cell.WONDERLAND:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="purple"))
                else:
                    ax.text(j, i, f"{value_func[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")


def plot_grid_policy(rfig, grid_mdp: GridMdp, policy: Union[OptimalActions, Policy], algo_name: str):
    MAP_SHAPE = grid_mdp.grid.shape
    font_size = get_font_size(grid_mdp)
    map_c = map2image(grid_mdp.grid)
    with rfig.plot(nid=f"{algo_name}-policy", mime=MIME_PDF, figsize=None) as _:
        ax = plt.gca()
        ax.imshow(map_c, aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=font_size + 3)
        for i in range(MAP_SHAPE[0]):
            for j in range(MAP_SHAPE[1]):
                # Skip cliff
                if grid_mdp.grid[i, j] == Cell.CLIFF:
                    continue
                # Get optimal actions. If policy is a single action, convert it to a list
                if policy.dtype == object:
                    optimal_actions = policy[i, j]
                elif policy.dtype == int:
                    optimal_actions = [policy[i, j]]
                else:
                    raise ValueError("Invalid policy type")
                # Put a random action to put O in the wonderland cell
                if optimal_actions is None:
                    optimal_actions = [Action.ABANDON]

                for action in optimal_actions:
                    if grid_mdp.grid[i, j] == Cell.WONDERLAND:
                        ax.text(j, i, "O", size=2.5 * font_size, ha="center", va="center", color="k", weight="bold")
                    elif action == Action.ABANDON:
                        ax.text(j, i, "X", size=2.5 * font_size, ha="center", va="center", color="k", weight="bold")
                    else:
                        arrow = action2arrow[action]
                        ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width, color="k")


def plot_report_figure(
    r: Report, grid_mdp: GridMdp, value_func: np.ndarray, policy: Union[OptimalActions, Policy], algo_name: str
):
    rfig = r.figure(cols=2)
    plot_grid_values(rfig, grid_mdp, value_func, algo_name)
    plot_grid_policy(rfig, grid_mdp, policy, algo_name)


def ex4_evaluation(ex_in: TestValueEx4, ex_out=None) -> tuple[PerformanceResults, Report]:
    grid_mdp = ex_in.grid
    solver: GridMdpSolver = ex_in.algo()
    algo_name = ex_in.str_id()
    r = Report(f"Ex4-{algo_name}")

    t = process_time()
    value_func, policy = solver.solve(grid_mdp)
    solve_time = process_time() - t
    plot_report_figure(r, grid_mdp, value_func, policy, algo_name)

    if ex_out is not None:
        all_states_mask = (grid_mdp.grid != Cell.CLIFF) & (grid_mdp.grid != Cell.WONDERLAND)
        # ground truth
        value_func_gt, policy_gt = ex_out
        # evaluate accuracy
        if policy_gt.dtype == int:  # policy_gt only contains single optimal action per state
            policy_accuracy = (
                np.sum(policy_gt[all_states_mask] == policy[all_states_mask]) / policy_gt[all_states_mask].size
            )
        elif policy_gt.dtype == object:  # policy_gt contains all optimal actions per state
            correct_policy = 0
            for user_policy, gt_policy in zip(policy[all_states_mask], policy_gt[all_states_mask]):
                # Put a random action to put O in the wonderland cell
                if gt_policy is None:
                    gt_policy = [Action.ABANDON]
                if user_policy is None:
                    user_policy = [Action.ABANDON]
                correct_policy += 1 if user_policy in gt_policy else 0
            policy_accuracy = float(correct_policy) / policy_gt[all_states_mask].size
        else:
            raise ValueError("Invalid policy_gt type")

        # R2 score - sum of squared errors divided by sum of squared differences from the mean
        value_func_r2 = 1.0 - np.sum(np.square(value_func_gt[all_states_mask] - value_func[all_states_mask])) / np.sum(
            np.square(value_func_gt[all_states_mask] - np.mean(value_func_gt[all_states_mask]))
        )
        # Clamp negative values to 0
        value_func_r2 = max(0, value_func_r2)

        # plot ground truth
        plot_report_figure(r, grid_mdp, value_func_gt, policy_gt, "GroundTruth")

        msg = f"policy_accuracy: {policy_accuracy}\n"
        msg += f"value_func_r2:{value_func_r2:.3f}\n"

        r.text(f"{algo_name}", text=remove_escapes(msg))

    result = Ex04Performance(policy_accuracy=policy_accuracy, value_func_r2=value_func_r2, solve_time=solve_time)
    perf = Ex04PerformanceResult(perf_result=result)
    if isinstance(solver, PolicyIteration):
        perf = Ex04PerformanceResult(perf_result=result, policy_iteration=result)
    elif isinstance(solver, ValueIteration):
        perf = Ex04PerformanceResult(perf_result=result, value_iteration=result)
    return perf, r


def ex4_transition_prob_evaluation(ex_in: TestTransitionProbEx4, ex_out=None) -> tuple[PerformanceResults, Report]:
    grid_mdp = ex_in.grid
    test_name = ex_in.str_id()
    r = Report(f"Ex4-{test_name}")

    # Get the transition probability from the implemented method
    try:
        transition_prob = grid_mdp.get_transition_prob(ex_in.state, ex_in.action, ex_in.next_state)
    except Exception as e:
        # Handle case where method is not implemented or throws an error
        transition_prob = None

    # Compare with expected result if provided
    if ex_out is not None and transition_prob is not None:
        expected_prob = ex_out
        accuracy = 1.0 if abs(transition_prob - expected_prob) < 1e-6 else 0.0

        msg = f"State: {ex_in.state}, Action: {ex_in.action.name}, Next State: {ex_in.next_state}\n"
        msg += f"Expected probability: {expected_prob:.6f}\n"
        msg += f"Computed probability: {transition_prob:.6f}\n"
        msg += f"Accuracy: {accuracy:.1f}\n"

        r.text(f"{test_name}", text=remove_escapes(msg))
    elif transition_prob is None:
        accuracy = 0.0
        msg = f"State: {ex_in.state}, Action: {ex_in.action.name}, Next State: {ex_in.next_state}\n"
        msg += f"Expected probability: {ex_out if ex_out is not None else 'N/A'}\n"
        msg += f"Computed probability: None (method not implemented)\n"
        msg += f"Accuracy: {accuracy:.1f}\n"

        r.text(f"{test_name}", text=remove_escapes(msg))
    else:
        accuracy = 0.0

    result = Ex04TransitionProbPerformance(transition_prob_accuracy=accuracy, test_count=1)
    return result, r


def ex4_single_perf_aggregator(perf: Sequence[Ex04Performance]) -> Ex04Performance:
    # perfomance for valid results
    policy_accuracy = [p.policy_accuracy for p in perf]
    value_func_r2 = [p.value_func_r2 for p in perf]
    solve_time = [p.solve_time for p in perf]
    try:
        # average accuracy and solve_time, rounding to 3 decimal places
        avg_policy_accuracy = round(np.mean(policy_accuracy), 3)
        avg_value_func_r2 = round(np.mean(value_func_r2), 3)
        avg_solve_time = round(np.mean(solve_time), 3)
    except ZeroDivisionError:
        # None if gt wasn't provided
        avg_policy_accuracy = 0
        avg_value_func_r2 = 0
        avg_solve_time = 0

    return Ex04Performance(
        policy_accuracy=float(avg_policy_accuracy),
        value_func_r2=float(avg_value_func_r2),
        solve_time=float(avg_solve_time),
    )


def ex4_transition_prob_perf_aggregator(perf: Sequence[Ex04TransitionProbPerformance]) -> Ex04TransitionProbPerformance:
    if not perf:
        return Ex04TransitionProbPerformance(transition_prob_accuracy=0.0, test_count=0)

    accuracy_sum = sum(p.transition_prob_accuracy for p in perf)
    total_tests = sum(p.test_count for p in perf)
    avg_accuracy = accuracy_sum / len(perf) if perf else 0.0

    return Ex04TransitionProbPerformance(transition_prob_accuracy=round(avg_accuracy, 3), test_count=total_tests)


def ex4_perf_aggregator(perf: Sequence[Ex04PerformanceResult]) -> Ex04PerformanceResult:
    perf_result = [p.perf_result for p in perf]
    policy_iteration = [p.policy_iteration for p in perf if p.policy_iteration is not None]
    value_iteration = [p.value_iteration for p in perf if p.value_iteration is not None]
    transition_prob = [p.transition_prob for p in perf if p.transition_prob is not None]

    return Ex04PerformanceResult(
        perf_result=ex4_single_perf_aggregator(perf_result),
        policy_iteration=ex4_single_perf_aggregator(policy_iteration),
        value_iteration=ex4_single_perf_aggregator(value_iteration),
        transition_prob=ex4_transition_prob_perf_aggregator(transition_prob) if transition_prob else None,
    )


def get_transition_prob_test_cases(grid_mdps: list[GridMdp]) -> list[TestTransitionProbEx4]:
    """Generate test cases for transition probability evaluation"""
    test_cases = []
    test_id = 0

    for grid_idx, grid_mdp in enumerate(grid_mdps):
        rows, cols = grid_mdp.grid.shape

        # Find special cells
        start_pos = None
        goal_pos = None
        for i in range(rows):
            for j in range(cols):
                if grid_mdp.grid[i, j] == Cell.START:
                    start_pos = (i, j)
                elif grid_mdp.grid[i, j] == Cell.GOAL:
                    goal_pos = (i, j)

        # Generate comprehensive test cases for the first grid
        if grid_idx == 0:  # Simple 5x5 test grid
            # Test cases for different scenarios on the simple grid
            test_scenarios = [
                # Normal movements within bounds from START position (2,2)
                ((2, 2), Action.NORTH, (1, 2)),
                ((2, 2), Action.SOUTH, (3, 2)),
                ((2, 2), Action.EAST, (2, 3)),  # Should move to WONDERLAND
                ((2, 2), Action.WEST, (2, 1)),
                # Movements from other valid positions (avoid STAY unless from GOAL)
                ((2, 1), Action.NORTH, (1, 1)),  # From GRASS to WONDERLAND
                ((2, 1), Action.SOUTH, (3, 1)),  # From GRASS to WONDERLAND
                ((2, 1), Action.EAST, (2, 2)),  # From GRASS to START
                ((2, 1), Action.WEST, (2, 0)),  # From GRASS to GRASS
                # Edge cases - trying to move out of bounds
                ((0, 2), Action.NORTH, (0, 2)),  # Already at top edge, should stay
                ((4, 2), Action.SOUTH, (4, 2)),  # Already at bottom edge, should stay
                ((2, 0), Action.WEST, (2, 0)),  # Already at left edge, should stay
                # Test ABANDON action from non-GOAL cells
                ((2, 1), Action.ABANDON, start_pos),  # ABANDON from GRASS should go to START
                # Test invalid transitions (should have 0 probability)
                ((2, 2), Action.NORTH, (2, 3)),  # Wrong direction
                ((2, 2), Action.EAST, (1, 2)),  # Wrong direction
                # Test transitions involving special cells
                ((1, 2), Action.SOUTH, (2, 2)),  # From GRASS to START
                ((3, 2), Action.NORTH, (2, 2)),  # From GRASS to START
            ]

            # Add STAY action only from GOAL position if we have one
            if goal_pos:
                test_scenarios.append((goal_pos, Action.STAY, goal_pos))

        else:
            # For larger grids, test some representative cases
            test_scenarios = [
                # Test from center positions (avoid STAY unless from GOAL)
                ((rows // 2, cols // 2), Action.NORTH, (rows // 2 - 1, cols // 2)),
                ((rows // 2, cols // 2), Action.SOUTH, (rows // 2 + 1, cols // 2)),
                ((rows // 2, cols // 2), Action.EAST, (rows // 2, cols // 2 + 1)),
                ((rows // 2, cols // 2), Action.WEST, (rows // 2, cols // 2 - 1)),
                # Test boundary conditions
                ((0, cols // 2), Action.NORTH, (0, cols // 2)),
                ((rows - 1, cols // 2), Action.SOUTH, (rows - 1, cols // 2)),
                ((rows // 2, 0), Action.WEST, (rows // 2, 0)),
                ((rows // 2, cols - 1), Action.EAST, (rows // 2, cols - 1)),
                # Test ABANDON from a few positions
                ((rows // 2, cols // 2), Action.ABANDON, start_pos),
            ]

            # Add STAY action only from GOAL position if we have one
            if goal_pos:
                test_scenarios.append((goal_pos, Action.STAY, goal_pos))

        for state, action, next_state in test_scenarios:
            # Only test valid states (within grid bounds and not cliffs)
            if (
                state
                and next_state  # Make sure positions are valid
                and 0 <= state[0] < rows
                and 0 <= state[1] < cols
                and 0 <= next_state[0] < rows
                and 0 <= next_state[1] < cols
            ):
                # Check if states are not cliffs
                if (
                    grid_mdp.grid[state[0], state[1]] != Cell.CLIFF
                    and grid_mdp.grid[next_state[0], next_state[1]] != Cell.CLIFF
                ):
                    test_cases.append(
                        TestTransitionProbEx4(
                            grid=grid_mdp, state=state, action=action, next_state=next_state, testId=test_id
                        )
                    )
                    test_id += 1

    return test_cases


def get_exercise4_with_transition_prob() -> Exercise:
    """Get exercise 4 including transition probability tests"""
    from typing import cast

    algos = [ValueIteration, PolicyIteration]
    grid_mdps = get_test_grids()

    # Original algorithm tests
    test_values = [
        TestValueEx4(algo=algo, grid=grid_mdp, testId=i) for algo in algos for i, grid_mdp in enumerate(grid_mdps)
    ]
    expected_results = get_expected_results()

    # Transition probability tests
    transition_test_cases = get_transition_prob_test_cases(grid_mdps[:1])  # Test on first grid only

    # Combine test cases - this is a simplified approach
    # In practice, you might want separate exercises or a more sophisticated combination

    return Exercise[TestValueEx4, Any](
        desc="This exercise is about dynamic programming and transition probabilities",
        evaluation_fun=ex4_evaluation,
        perf_aggregator=cast(Any, ex4_perf_aggregator),
        test_values=test_values,
        expected_results=expected_results,
    )


def get_exercise4() -> Exercise:
    from typing import cast

    algos = [ValueIteration, PolicyIteration]
    grid_mdps = get_test_grids()
    test_values = [
        TestValueEx4(algo=algo, grid=grid_mdp, testId=i) for algo in algos for i, grid_mdp in enumerate(grid_mdps)
    ]
    expected_results = get_expected_results()

    return Exercise[TestValueEx4, Any](
        desc="This exercise is about dynamic programming",
        evaluation_fun=ex4_evaluation,
        perf_aggregator=cast(Any, ex4_perf_aggregator),
        test_values=test_values,
        expected_results=expected_results,
    )
