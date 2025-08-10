from pathlib import Path
from dataclasses import dataclass

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
            [Cell.WONDERLAND, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.WONDERLAND],
            [Cell.GRASS, Cell.GRASS, Cell.START, Cell.GRASS, Cell.GOAL],
            [Cell.WONDERLAND, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.WONDERLAND],
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
    test_maps.append(generate_map(MAP_SHAPE_2, swamp_ratio, n_wonderland=4, n_cliff=10, n_seed=5))
    test_maps.append(generate_map(MAP_SHAPE_3, swamp_ratio, n_wonderland=5, n_cliff=15, n_seed=110))

    # additional maps for evaluation
    for map_info in evaluation_tests:
        test_maps.append(
            generate_map(map_info[0], swamp_ratio, n_wonderland=map_info[1], n_cliff=map_info[2], n_seed=map_info[3])
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

    # Get the results array - this should contain all transition probability results
    # in the same order as the test cases are generated
    transition_probs = all_data["transition_probs"].tolist()

    # Return the first len(test_cases) results
    # (assuming test cases are generated in the same order as when we computed the ground truth)
    return transition_probs[: len(test_cases)]


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
