from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp
from pdm4ar.exercises.ex04.structures import OptimalActions, ValueFunc, Cell, Action, State
from pdm4ar.exercises_def.ex04.map import generate_map, random_map
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


SMALL_TEST_MAP_SPECS: list[tuple[tuple[int, int], int]] = [((6, 6), 1), ((9, 9), 2), ((12, 12), 1)]
"""The three optional extra test maps (ids 3-5), regenerated deterministically."""


def get_small_test_grids() -> list[GridMdp]:
    """The optional extra maps (ids 3-5), enabled with ALL_MAPS in ex04.py."""
    return [GridMdp(grid=random_map(shape, seed=seed), gamma=0.9) for shape, seed in SMALL_TEST_MAP_SPECS]


def get_test_grids(evaluation_tests: list[tuple[tuple[int, int], int, int]] = []) -> list[GridMdp]:
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


def get_expected_results_algo(map_ids: tuple = (0, 1, 2)) -> list[tuple[ValueFunc, OptimalActions]]:
    """Solutions for the given map ids (0-2: public maps, 3-5: the optional extra maps)."""
    data_dir = Path(__file__).parent
    all_data = np.load(data_dir / "data/expected_results.npz", allow_pickle=True)
    one_pass = [(all_data[f"value_func_{mi}"], all_data[f"policy_{mi}"]) for mi in map_ids]
    # once for ValueIteration, once for PolicyIteration
    return one_pass + one_pass


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


def get_test_mdps_aug(map_ids: tuple = (0, 1, 2)) -> list[tuple[str, int, AugmentedGridMdp]]:
    """(case_name, map_id, mdp) over the requested maps (ids 0-2: the public
    maps, ids 3-5: the optional extra maps)."""
    grids_by_id = get_test_grids() + (get_small_test_grids() if max(map_ids) > 2 else [])
    maps = [(mi, grids_by_id[mi].grid) for mi in map_ids]
    out = []
    for case_name, cls in AUG_CASES:
        for mi, grid in maps:
            out.append((case_name, mi, cls(grid=grid, gamma=0.9)))
    return out


def get_expected_results_algo_aug(map_ids: tuple = (0, 1, 2)) -> list[tuple[ValueFunc, OptimalActions]]:
    """Aligned with get_exercise4's Part-2 test order: every (case, map)
    pair once for ValueIteration, once for PolicyIteration."""
    data_dir = Path(__file__).parent
    data = np.load(data_dir / "data/expected_results_aug.npz", allow_pickle=True)
    one_pass = [
        (data[f"{case_name}_value_{mi}"], data[f"{case_name}_policy_{mi}"])
        for case_name, _ in AUG_CASES
        for mi in map_ids
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


# Probes on the 5x5 map; expected values come from the full-coverage
# data/expected_transition_results_aug.npz, so you can add your own probes.
AUG_PROBES = [
    ("forecast", (2, 1, 0), Action.EAST, (2, 2, 0)),
    ("forecast", (2, 1, 1), Action.EAST, (2, 2, 1)),
    ("forecast", (1, 2, 0), Action.NORTH, (0, 2, 1)),
    ("forecast", (3, 3, 1), Action.WEST, (3, 3, 0)),
    ("forecast", (1, 2, 1), Action.ABANDON, (2, 2, 0)),
    ("momentum", (2, 1, 0), Action.EAST, (2, 2, 4)),
    ("momentum", (2, 1, 4), Action.EAST, (2, 2, 4)),
    ("momentum", (2, 1, 2), Action.EAST, (2, 1, 0)),
    ("momentum", (2, 1, 2), Action.EAST, (2, 0, 2)),
    ("momentum", (2, 3, 1), Action.EAST, (1, 3, 1)),
    ("momentum", (3, 3, 1), Action.WEST, (3, 3, 0)),
    ("glitch", (2, 1, 0), Action.EAST, (2, 2, 0)),
    ("glitch", (2, 1, 1), Action.EAST, (2, 2, 1)),
    ("glitch", (2, 1, 0), Action.EAST, (2, 2, 1)),
    ("glitch", (3, 1, 1), Action.NORTH, (2, 2, 0)),
]


def get_transition_prob_test_cases_aug() -> list[TestTransitionProbAug]:
    grid = get_simple_test_grid()
    classes = dict(AUG_CASES)
    cases = []
    counters: dict = {}
    for case_name, state, action, next_state in AUG_PROBES:
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


def get_expected_results_transition_aug(test_cases: list[TestTransitionProbAug]) -> list[float]:
    """Load pre-computed Part-2 transition probabilities for the given test
    cases, mirroring the Part-1 mechanism."""
    data_dir = Path(__file__).parent
    all_data = np.load(data_dir / "data/expected_transition_results_aug.npz", allow_pickle=True)
    transition_probs: dict = all_data["transition_probs"].item()
    return [
        transition_probs[(test.case_name, test.state, test.action, test.next_state)]
        for test in test_cases
    ]
