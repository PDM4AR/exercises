from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp
from pdm4ar.exercises.ex04.structures import OptimalActions, ValueFunc, Cell
from pdm4ar.exercises_def.ex04.map import generate_map

if TYPE_CHECKING:
    from pdm4ar.exercises_def.ex04.ex04 import TestTransitionProbEx4


def get_simple_test_grid() -> np.ndarray:
    simple_map = np.array(
        [
            [Cell.CLIFF, Cell.CLIFF, Cell.GRASS, Cell.CLIFF, Cell.CLIFF],
            [Cell.CLIFF, Cell.WONDERLAND, Cell.GRASS, Cell.SWAMP, Cell.CLIFF],
            [Cell.GRASS, Cell.GRASS, Cell.START, Cell.WONDERLAND, Cell.GOAL],
            [Cell.CLIFF, Cell.WONDERLAND, Cell.GRASS, Cell.SWAMP, Cell.CLIFF],
            [Cell.CLIFF, Cell.CLIFF, Cell.GRASS, Cell.CLIFF, Cell.CLIFF],
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


def get_expected_results_transition(test_cases: list["TestTransitionProbEx4"]) -> list[float]:
    """Load pre-computed transition probability results for the given test cases"""
    data_dir = Path(__file__).parent
    all_data = np.load(data_dir / "data/expected_transition_results.npz", allow_pickle=True)
    
    # Get the results array - this should contain all transition probability results
    # in the same order as the test cases are generated
    transition_probs = all_data["transition_probs"].tolist()
    
    # Return the first len(test_cases) results 
    # (assuming test cases are generated in the same order as when we computed the ground truth)
    return transition_probs[:len(test_cases)]


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
