from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp
from pdm4ar.exercises.ex04.structures import AllOptimalActions, ValueFunc, Cell
from pdm4ar.exercises_def.ex04.map import generate_map


def get_simple_test_grid() -> np.ndarray:
    simple_map = np.array(
        [
            [Cell.CLIFF, Cell.CLIFF, Cell.GRASS, Cell.CLIFF, Cell.CLIFF],
            [Cell.CLIFF, Cell.WORMHOLE, Cell.GRASS, Cell.SWAMP, Cell.CLIFF],
            [Cell.GRASS, Cell.GRASS, Cell.START, Cell.WORMHOLE, Cell.GOAL],
            [Cell.CLIFF, Cell.WORMHOLE, Cell.GRASS, Cell.SWAMP, Cell.CLIFF],
            [Cell.CLIFF, Cell.CLIFF, Cell.GRASS, Cell.CLIFF, Cell.CLIFF],
        ]
    )
    return simple_map


def get_test_grids(evaluation_tests: list[tuple[tuple[int, int], int, int]] = [], n_eval_seed=12) -> List[GridMdp]:
    MAP_SHAPE_2 = (10, 10)
    MAP_SHAPE_3 = (40, 40)

    test_maps = []
    swamp_ratio = 0.2
    test_maps.append(get_simple_test_grid())
    test_maps.append(generate_map(MAP_SHAPE_2, swamp_ratio, n_wormhole=4, n_cliff=10, n_seed=5))
    test_maps.append(generate_map(MAP_SHAPE_3, swamp_ratio, n_wormhole=5, n_cliff=15, n_seed=110))

    # additional maps for evaluation
    for map_info in evaluation_tests:
        test_maps.append(
            generate_map(map_info[0], swamp_ratio, n_wormhole=map_info[1], n_cliff=map_info[2], n_seed=n_eval_seed)
        )

    discount = 0.9
    data_in: List[GridMdp] = []
    for m in test_maps:
        p = GridMdp(grid=m, gamma=discount)
        data_in.append(p)

    return data_in


def get_expected_results() -> List[Tuple[ValueFunc, AllOptimalActions]]:
    # fmt: off
    expected_results = defaultdict(dict)

    value_func1 = np.array([[146.06783255, 153.61600686, 176.32872787, 205.63826482, 242.16355181, 287.65529141, 342.69300758, 399.81133988, 339.16059923, 252.57964034],
                            [146.06783254, 166.22278297, 197.72505653, 237.07857855, 285.34721196, 344.62034089, 414.75657905, 499.99999994, 413.42067606, 331.26673385],
                            [146.06783255, 150.42775799, 173.40870282, 202.8928186 , 237.13762815, 302.45949732, 356.10770642, 415.63767263, 355.18168175, 294.55980139],
                            [146.06783255, 146.06783254, 151.26118937, 173.84498043, 218.99583189, 259.39930451, 303.48095606, 343.87958113, 300.36955925, 250.59789065],
                            [146.06783255, 146.06783255, 146.06783255, 161.86423192, 187.70140973, 219.20679346, 251.4978377 , 253.80484639, 227.14818271, 194.45511571]])
    
    policy1 = np.array([[5., 3., 3., 3., 3., 2., 2., 2., 2., 1.],
                        [5., 3., 3., 3., 3., 3., 3., 4., 1., 1.],
                        [5., 3., 3., 3., 3., 3., 3., 0., 1., 1.],
                        [5., 5., 3., 3., 3., 3., 0., 0., 0., 1.],
                        [5., 5., 5., 3., 3., 0., 0., 0., 0., 0.]])
    
    value_func2 = np.array([[189.28148956, 189.41251223, 224.67466313, 258.79253367, 287.39709725, 256.69285849, 223.14878486, 196.19317862, 189.28148956, 189.28148956],
                            [196.74824033, 223.74690619, 261.16418848, 304.61288026, 346.61071907, 301.67179239, 258.38351794, 221.3090621 , 189.28148956, 189.28148956],
                            [223.32679293, 259.98696704, 304.72089594, 356.47455136, 415.68086864, 352.65964479, 298.94292747, 253.1944983 , 214.87246442, 189.28148956],
                            [244.08019587, 290.26446462, 346.85649482, 415.70141186, 499.99999999, 373.71939294, 313.72421366, 262.59210352, 220.98606187, 192.33127266],
                            [221.71620422, 257.75057867, 301.78853032, 352.68777866, 373.74243122, 320.09384467, 271.31328142, 216.86289713, 189.28148956, 189.28148956],
                            [195.40121943, 221.93027397, 258.38332285, 298.99338063, 313.94796503, 272.53670703, 219.41203941, 191.76006732, 189.28148956, 189.28148956],
                            [189.28148956, 193.80194916, 221.42387729, 253.39044594, 264.09423173, 232.95148722, 201.09038026, 189.28148956, 189.28148956, 189.28148956],
                            [189.28148956, 189.28148956, 193.00939644, 215.41407781, 222.71613805, 201.33818766, 189.28148956, 189.28148956, 189.28148956, 189.28148956],
                            [189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956],
                            [189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956, 189.28148956]])
    
    policy2 = np.array([[5., 3., 2., 2., 2., 2., 2., 1., 5., 5.],
                        [3., 3., 2., 2., 2., 2., 1., 1., 5., 5.],
                        [3., 3., 3., 2., 2., 1., 1., 1., 1., 5.],
                        [3., 3., 3., 3., 4., 1., 1., 1., 1., 1.],
                        [3., 3., 3., 0., 0., 1., 1., 1., 5., 5.],
                        [3., 3., 0., 0., 0., 0., 1., 1., 5., 5.],
                        [5., 0., 0., 0., 0., 0., 1., 5., 5., 5.],
                        [5., 5., 0., 0., 0., 0., 5., 5., 5., 5.],
                        [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                        [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.]])


    expected_results = [(value_func1, policy1),
                        (value_func2, policy2),
                        (value_func1, policy1),
                        (value_func2, policy2)]
    # fmt: on
    return expected_results
