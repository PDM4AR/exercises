from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.map import generate_map


def get_test_grids(evaluation_tests: List[Tuple[int, int]] = [], n_seed=3) -> List[GridMdp]:
    MAP_SHAPE_1 = (5, 5)
    MAP_SHAPE_2 = (5, 10)
    
    test_maps = []
    swamp_ratio = 0.2
    for ms in [MAP_SHAPE_1, MAP_SHAPE_2]:
        test_maps.append(generate_map(ms, swamp_ratio, n_seed=n_seed))

    # additional maps for evaluation
    for ms in evaluation_tests:
        test_maps.append(generate_map(ms, swamp_ratio, n_seed=n_seed))

    discount = 0.9
    data_in: List[GridMdp] = []
    for m in test_maps:
        p = GridMdp(grid=m, gamma=discount)
        data_in.append(p)

    return data_in

def get_expected_results() -> List[Tuple[ValueFunc, Policy]]:

    expected_results = defaultdict(dict)

    value_func1 = np.array([[ 39.84265676,  43.14174315,  49.29853516,  55.78348311,  47.5861003 ],
                            [ 41.94868662,  47.85491111,  57.17219093,  67.00625353,  58.04363969],
                            [ 44.94010729,  51.75866672,  67.94105695,  81.75008149,  68.38710939],
                            [ 47.58588583,  61.06486263,  74.53250172, 100.        ,  74.19489449],
                            [ 49.88601287,  57.39443376,  67.82993153,  80.95925489,  67.78794278]])
    
    policy1 = np.array([[3., 3., 2., 2., 2.],
                        [3., 3., 2., 2., 2.],
                        [3., 3., 3., 2., 1.],
                        [3., 3., 3., 4., 1.],
                        [3., 3., 3., 0., 1.]])
    
    value_func2 = np.array([[22.74225862, 23.85248502, 26.39210492, 29.51345029, 33.19595621, 41.8715872 ,49.02132618 , 54.89143809 , 44.73554115 ,  42.51010951],
                            [21.32182461, 22.47862857, 27.13789263, 33.0736577 , 40.33193869, 48.61663553, 57.95237278,  66.98285241,  57.15262368,  48.68510902],
                            [21.84873653, 23.63303761, 28.63824906, 34.67664499, 47.27318236, 57.78025977, 69.04564703,  81.82684419,  68.32942371,  56.96776226],
                            [22.01402187, 26.36889492, 32.31591917, 39.64099745, 48.61151001, 66.43615598, 81.76773042,  99.99999996,  73.83033269,  60.63693888], 
                            [25.46615114, 28.42826702, 32.43613467, 37.38381193, 43.45448795, 56.84740304, 68.08793643,  79.79887354,  60.61374219,  51.76845974]])
    
    policy2 = np.array([[3., 3., 3., 3., 3., 3., 2., 2., 2., 2.],
                        [0., 3., 3., 3., 3., 3., 2., 2., 2., 1.],
                        [3., 3., 3., 3., 3., 3., 3., 2., 1., 1.],
                        [3., 3., 3., 3., 3., 3., 3., 4., 1., 1.],
                        [3., 3., 3., 3., 3., 3., 0., 0., 1., 0.]])

    expected_results[0] = [(value_func1, policy1), (value_func2, policy2)]
    expected_results[1] = [(value_func1, policy1), (value_func2, policy2)]
    
    return expected_results
