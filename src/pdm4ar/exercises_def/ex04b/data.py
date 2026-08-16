from dataclasses import dataclass
from pathlib import Path
from typing import Type

import numpy as np

from pdm4ar.exercises.ex04.structures import Cell
from pdm4ar.exercises.ex04b.mdp import (AugmentedGridMdp, FogGridMdp,
                                        GlitchGridMdp, MomentumGridMdp)
from pdm4ar.exercises.ex04b.structures import Action, Policy, State, ValueFunc
from pdm4ar.exercises_def import ExIn
from pdm4ar.exercises_def.ex04.map import generate_map

CASES: list[tuple[str, Type[AugmentedGridMdp]]] = [
    ("momentum", MomentumGridMdp),
    ("forecast", FogGridMdp),
    ("glitch", GlitchGridMdp),
]


def get_simple_test_grid_ex04b() -> np.ndarray:
    """The ex04 5x5 doc example with its WONDERLAND cells turned to GRASS."""
    return np.array(
        [
            [Cell.CLIFF, Cell.GRASS, Cell.GRASS, Cell.GRASS, Cell.CLIFF],
            [Cell.GRASS, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.GRASS],
            [Cell.GRASS, Cell.GRASS, Cell.START, Cell.GRASS, Cell.GOAL],
            [Cell.GRASS, Cell.SWAMP, Cell.GRASS, Cell.SWAMP, Cell.GRASS],
            [Cell.CLIFF, Cell.GRASS, Cell.GRASS, Cell.GRASS, Cell.CLIFF],
        ],
        dtype=int,
    )


def get_test_maps_ex04b() -> list[np.ndarray]:
    """Same generator and seeds as ex04, with the wonderland removed."""
    return [
        get_simple_test_grid_ex04b(),
        generate_map((10, 10), 0.2, n_wonderland=0, n_cliff=10, n_seed=5),
        generate_map((40, 40), 0.2, n_wonderland=0, n_cliff=15, n_seed=110),
    ]


def get_test_mdps_ex04b() -> list[tuple[str, int, AugmentedGridMdp]]:
    """(case_name, map_index, mdp) in the canonical grading order."""
    maps = get_test_maps_ex04b()
    out = []
    for case_name, cls in CASES:
        for mi, grid in enumerate(maps):
            out.append((case_name, mi, cls(grid=grid, gamma=0.9)))
    return out


def get_expected_results_algo_ex04b() -> list[tuple[ValueFunc, Policy]]:
    """Ground truth aligned with get_exercise4b's test order: the 9
    (case, map) pairs once for ValueIteration, once for PolicyIteration."""
    data_dir = Path(__file__).parent
    data = np.load(data_dir / "data/expected_results_ex04b.npz", allow_pickle=True)
    one_pass = [
        (data[f"{case_name}_value_{mi}"], data[f"{case_name}_policy_{mi}"])
        for case_name, _ in CASES
        for mi in range(3)
    ]
    return one_pass + one_pass


# -------------------------------------------------------- transition probes
@dataclass
class TestTransitionProbEx4b(ExIn):
    mdp: AugmentedGridMdp
    case_name: str
    state: State
    action: Action
    next_state: State
    testId: int = 0

    def str_id(self) -> str:
        return f"TransitionProb-{self.case_name}{self.testId}"


# (case, (i, j, z), action, (i', j', z'), expected probability)
# Computed from the validated reference implementation on the 5x5 map.
PROBES = [
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


def get_transition_prob_test_cases_ex04b() -> list[TestTransitionProbEx4b]:
    grid = get_simple_test_grid_ex04b()
    classes = dict(CASES)
    cases = []
    counters: dict[str, int] = {}
    for case_name, state, action, next_state, _ in PROBES:
        tid = counters.get(case_name, 0)
        counters[case_name] = tid + 1
        cases.append(
            TestTransitionProbEx4b(
                mdp=classes[case_name](grid=grid, gamma=0.9),
                case_name=case_name,
                state=state,
                action=action,
                next_state=next_state,
                testId=tid,
            )
        )
    return cases


def get_expected_results_transition_ex04b() -> list[float]:
    return [expected for *_ignored, expected in PROBES]
