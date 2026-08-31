from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pickle

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp
from pdm4ar.exercises.ex04.structures import OptimalActions, ValueFunc, Cell, Action, State
from pdm4ar.exercises_def.ex04.map import generate_map
from pdm4ar.exercises_def import ExIn


@dataclass
class TestTransitionProbEx4(ExIn):
    grid: GridMdp
    state: State
    action: Action
    next_state: State
    testId: int = 0

    def str_id(self) -> str:
        return f"TransitionProb{self.testId}_s{self.state}_a{self.action.name}_ns{self.next_state}"


def get_simple_test_grid() -> np.ndarray:
    simple_map = np.array(
        [
            [Cell.CLIFF, Cell.GRASS, Cell.GRASS, Cell.GRASS, Cell.CLIFF],
            [Cell.GRASS, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.GRASS],
            [Cell.GRASS, Cell.GRASS, Cell.START, Cell.GRASS, Cell.GOAL],
            [Cell.GRASS, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.GRASS],
            [Cell.CLIFF, Cell.GRASS, Cell.GRASS, Cell.GRASS, Cell.CLIFF],
        ]
    )
    return simple_map


def get_test_grids(evaluation_tests: list[tuple[tuple[int, int], int, int, int]] = []) -> list[GridMdp]:
    MAP_SHAPE_2 = (10, 10)
    MAP_SHAPE_3 = (40, 40)

    test_maps = []
    swamp_ratio = 0.2
    test_maps.append(get_simple_test_grid())
    test_maps.append(generate_map(MAP_SHAPE_2, swamp_ratio, n_cliff=10, n_seed=5))
    test_maps.append(generate_map(MAP_SHAPE_3, swamp_ratio, n_cliff=15, n_seed=110))

    # additional maps for evaluation
    for map_info in evaluation_tests:
        test_maps.append(
            generate_map(map_info[0], swamp_ratio, n_cliff=map_info[1], n_seed=map_info[2])
        )

    discount = 0.9
    data_in: list[GridMdp] = []
    for m in test_maps:
        p = GridMdp(grid=m, gamma=discount)
        data_in.append(p)

    return data_in


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
                ((2, 2), Action.NORTH, (1, 2)),
                ((2, 2), Action.SOUTH, (3, 2)),
                ((2, 2), Action.EAST, (2, 3)),
                ((2, 2), Action.WEST, (2, 1)),
                ((2, 2), Action.EAST, (1, 3)),
                ((2, 1), Action.SOUTH, (3, 1)),
                ((2, 1), Action.EAST, (2, 2)),
                ((2, 1), Action.WEST, (2, 0)),
                ((2, 1), Action.ABANDON, start_pos),
                ((2, 2), Action.NORTH, (2, 3)),
                ((2, 2), Action.EAST, (1, 2)),
                ((1, 2), Action.SOUTH, (2, 2)),
                ((3, 2), Action.NORTH, (2, 2)),
                ((2, 0), Action.NORTH, (1, 1)),
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


def get_expected_results_transition(test_cases: list[TestTransitionProbEx4]) -> list[float]:
    """Load pre-computed transition probability results for the given test cases"""
    data_dir = Path(__file__).parent
    all_data = np.load(data_dir / "data/expected_transition_results.npz", allow_pickle=True)

    transition_probs: dict = all_data["transition_probs"].item()

    res: list[float] = []
    for test in test_cases:
        state = test.state
        action = test.action
        next_state = test.next_state
        res.append(transition_probs[(state, action, next_state)])

    return res


def get_expected_results_algo() -> list[tuple[ValueFunc, OptimalActions]]:
    data_dir = Path(__file__).parent
    all_data = np.load(data_dir / "data/expected_results.npz", allow_pickle=True)

    value_func_0 = all_data["value_func_0"]
    policy_0 = all_data["policy_0"]

    value_func_1 = all_data["value_func_1"]
    policy_1 = all_data["policy_1"]

    value_func_2 = all_data["value_func_2"]
    policy_2 = all_data["policy_2"]

    expected_results = [
        (value_func_0, policy_0),
        (value_func_1, policy_1),
        (value_func_2, policy_2),
        (value_func_0, policy_0),
        (value_func_1, policy_1),
        (value_func_2, policy_2),
    ]

    return expected_results


def load_transition_matrix() -> Dict:
    """Load the complete transition matrix for the first grid"""
    data_dir = Path(__file__).parent
    try:
        with open(data_dir / "data/expected_transition_matrix.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Transition matrix file not found. Run generate_expected_results.py to create it.")


def get_transition_probability(state: State, action: Action, next_state: State) -> float:
    """
    Get the transition probability P(s'|s,a) for the first grid.

    Args:
        state: Current state (row, col)
        action: Action taken
        next_state: Next state (row, col)

    Returns:
        Transition probability (0.0 if transition is not possible)
    """
    matrix_data = load_transition_matrix()
    transition_matrix = matrix_data["transition_matrix"]

    if (
        state in transition_matrix
        and action in transition_matrix[state]
        and next_state in transition_matrix[state][action]
    ):
        return transition_matrix[state][action][next_state]
    else:
        return 0.0


def get_all_transitions_from_state(state: State) -> Dict[Action, Dict[State, float]]:
    """
    Get all possible transitions from a given state.

    Args:
        state: Current state (row, col)

    Returns:
        Dictionary mapping actions to {next_state: probability} dictionaries
    """
    matrix_data = load_transition_matrix()
    transition_matrix = matrix_data["transition_matrix"]

    if state in transition_matrix:
        return transition_matrix[state]
    else:
        return {}


def get_valid_actions_for_state(state: State) -> list[Action]:
    """
    Get all valid actions from a given state.

    Args:
        state: Current state (row, col)

    Returns:
        List of valid actions from this state
    """
    transitions = get_all_transitions_from_state(state)
    return list(transitions.keys())


def get_possible_next_states(state: State, action: Action) -> Dict[State, float]:
    """
    Get all possible next states and their probabilities for a given state-action pair.

    Args:
        state: Current state (row, col)
        action: Action taken

    Returns:
        Dictionary mapping next_states to their transition probabilities
    """
    transitions = get_all_transitions_from_state(state)
    if action in transitions:
        return transitions[action]
    else:
        return {}


def get_grid_info() -> Dict:
    """
    Get basic information about the grid.

    Returns:
        Dictionary with grid_shape, start_pos, goal_pos
    """
    matrix_data = load_transition_matrix()
    return {
        "grid_shape": matrix_data["grid_shape"],
        "start_pos": matrix_data["start_pos"],
        "goal_pos": matrix_data["goal_pos"],
    }


# ---------------------------------------------------------------------------
# Part 2: augmented cases (momentum / forecast / glitch)
# ---------------------------------------------------------------------------
from typing import Type  # noqa: E402

from pdm4ar.exercises.ex04.mdp import (AugmentedGridMdp, FogGridMdp,  # noqa: E402
                                       GlitchGridMdp, MomentumGridMdp)
from pdm4ar.exercises.ex04.structures import AugmentedState  # noqa: E402

AUG_CASES: list[tuple[str, Type[AugmentedGridMdp]]] = [
    ("forecast", FogGridMdp),
    ("momentum", MomentumGridMdp),
    ("glitch", GlitchGridMdp),
]


def get_test_mdps_aug() -> list[tuple[str, int, AugmentedGridMdp]]:
    """(case_name, map_index, mdp) over the same maps as Part 1."""
    maps = [g.grid for g in get_test_grids()]
    out = []
    for case_name, cls in AUG_CASES:
        for mi, grid in enumerate(maps):
            out.append((case_name, mi, cls(grid=grid, gamma=0.9)))
    return out


def get_expected_results_algo_aug() -> list[tuple[ValueFunc, OptimalActions]]:
    """Aligned with get_exercise4's Part-2 test order: the 9 (case, map)
    pairs once for ValueIteration, once for PolicyIteration."""
    data_dir = Path(__file__).parent
    data = np.load(data_dir / "data/expected_results_aug.npz", allow_pickle=True)
    one_pass = [
        (data[f"{case_name}_value_{mi}"], data[f"{case_name}_policy_{mi}"])
        for case_name, _ in AUG_CASES
        for mi in range(3)
    ]
    return one_pass + one_pass


@dataclass
class TestTransitionProbAug(ExIn):
    mdp: AugmentedGridMdp
    case_name: str
    state: AugmentedState
    action: Action
    next_state: AugmentedState
    testId: int = 0

    def str_id(self) -> str:
        return f"TransitionProb-{self.case_name}{self.testId}"


# (case, (i, j, z), action, (i', j', z'), expected probability)
# Computed from the validated reference implementation on the 5x5 map.
AUG_PROBES = [
    ("forecast", (2, 1, 0), Action.EAST, (2, 2, 0), 0.5250000000),
    ("forecast", (2, 1, 1), Action.EAST, (2, 2, 1), 0.1650000000),
    ("forecast", (1, 2, 0), Action.NORTH, (0, 2, 1), 0.2250000000),
    ("forecast", (3, 3, 1), Action.WEST, (3, 3, 0), 0.1400000000),
    ("forecast", (1, 2, 1), Action.ABANDON, (2, 2, 0), 0.7000000000),
    ("momentum", (2, 1, 0), Action.EAST, (2, 2, 4), 0.7500000000),
    ("momentum", (2, 1, 4), Action.EAST, (2, 2, 4), 0.8500000000),
    ("momentum", (2, 1, 2), Action.EAST, (2, 1, 0), 0.0000000000),
    ("momentum", (2, 1, 2), Action.EAST, (2, 0, 2), 0.2500000000),
    ("momentum", (2, 3, 1), Action.EAST, (1, 3, 1), 0.1500000000),
    ("momentum", (3, 3, 1), Action.WEST, (3, 3, 0), 0.2000000000),
    ("glitch", (2, 1, 0), Action.EAST, (2, 2, 0), 0.6750000000),
    ("glitch", (2, 1, 1), Action.EAST, (2, 2, 1), 0.4200000000),
    ("glitch", (2, 1, 0), Action.EAST, (2, 2, 1), 0.0750000000),
    ("glitch", (3, 1, 1), Action.NORTH, (2, 2, 0), 0.0500000000),
]


# ---------------------------------------------------------------------------
# Practice maps: extra ungraded maps that run in the same built-in flow as the
# public test maps. They regenerate deterministically from (shape, seed), so
# only their solutions ship in data/expected_results_practice.npz.
# ---------------------------------------------------------------------------
PRACTICE_MAP_SPECS: list[tuple[tuple[int, int], int]] = [((6, 6), 1), ((9, 9), 2), ((12, 12), 1)]


def get_practice_grids() -> list[np.ndarray]:
    from pdm4ar.exercises_def.ex04.selfcheck import random_map

    return [random_map(shape, seed=seed) for shape, seed in PRACTICE_MAP_SPECS]


def get_expected_results_practice() -> list[tuple[ValueFunc, OptimalActions]]:
    """Aligned with get_exercise4's practice test order: base on every
    practice map, then each Part-2 case on every practice map, once for
    ValueIteration and once for PolicyIteration."""
    data_dir = Path(__file__).parent
    data = np.load(data_dir / "data/expected_results_practice.npz", allow_pickle=True)
    cases = ["base"] + [case_name for case_name, _ in AUG_CASES]
    one_pass = [
        (data[f"practice_{case_name}_value_{k}"], data[f"practice_{case_name}_policy_{k}"])
        for case_name in cases
        for k in range(len(PRACTICE_MAP_SPECS))
    ]
    return one_pass + one_pass


def get_transition_prob_test_cases_aug() -> list[TestTransitionProbAug]:
    grid = get_simple_test_grid()
    classes = dict(AUG_CASES)
    cases = []
    counters: dict = {}
    for case_name, state, action, next_state, _ in AUG_PROBES:
        tid = counters.get(case_name, 0)
        counters[case_name] = tid + 1
        cases.append(
            TestTransitionProbAug(
                mdp=classes[case_name](grid=grid, gamma=0.9),
                case_name=case_name,
                state=state,
                action=action,
                next_state=next_state,
                testId=tid,
            )
        )
    return cases


def get_expected_results_transition_aug() -> list[float]:
    return [expected for *_ignored, expected in AUG_PROBES]
