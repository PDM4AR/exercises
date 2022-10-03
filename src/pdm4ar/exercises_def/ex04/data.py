from typing import List

from pdm4ar.exercises_def.ex04.map import generate_map
from pdm4ar.exercises.ex04.mdp import GridMdp


def get_test_grids() -> List[GridMdp]:
    MAP_SHAPE_1 = (5, 5)
    MAP_SHAPE_2 = (5, 10)
    MAP_SHAPE_3 = (20, 20)
    MAP_SHAPE_4 = (30, 30)

    test_maps = []
    swamp_ratio = 0.2
    for ms in [MAP_SHAPE_1, MAP_SHAPE_2, MAP_SHAPE_3, MAP_SHAPE_4]:
        test_maps.append(generate_map(ms, swamp_ratio))

    discount = 0.8
    data_in: List[GridMdp] = []
    for m in test_maps:
        p = GridMdp(grid=m, gamma=discount)
        data_in.append(p)

    return data_in
