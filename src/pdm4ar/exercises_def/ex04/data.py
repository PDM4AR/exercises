from typing import List, Tuple
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc

from pdm4ar.exercises_def.ex04.map import generate_map
from pdm4ar.exercises.ex04.mdp import GridMdp


def get_test_grids(evaluation_tests: List[Tuple[int, int]] = []) -> List[GridMdp]:
    MAP_SHAPE_1 = (5, 5)
    MAP_SHAPE_2 = (5, 10)
    
    test_maps = []
    swamp_ratio = 0.2
    for ms in [MAP_SHAPE_1, MAP_SHAPE_2]:
        test_maps.append(generate_map(ms, swamp_ratio))

    # additiomal maps for evaluation
    for ms in evaluation_tests:
        test_maps.append(generate_map(ms, swamp_ratio))

    discount = 0.9
    data_in: List[GridMdp] = []
    for m in test_maps:
        p = GridMdp(grid=m, gamma=discount)
        data_in.append(p)

    return data_in

def get_expected_results() -> List[ValueFunc, Policy]:
    pass
