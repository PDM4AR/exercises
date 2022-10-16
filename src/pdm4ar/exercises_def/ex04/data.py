import imp
from inspect import ismethoddescriptor
import numpy as np
from typing import List, Tuple
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from collections import defaultdict


from pdm4ar.exercises_def.ex04.map import generate_map
from pdm4ar.exercises.ex04.mdp import GridMdp


def get_test_grids(evaluation_tests: List[Tuple[int, int]] = [], n_seed=3) -> List[GridMdp]:
    MAP_SHAPE_1 = (5, 5)
    MAP_SHAPE_2 = (5, 10)
    
    test_maps = []
    swamp_ratio = 0.2
    for ms in [MAP_SHAPE_1, MAP_SHAPE_2]:
        test_maps.append(generate_map(ms, swamp_ratio, n_seed=n_seed))

    # additiomal maps for evaluation
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

    value_func1 = np.array([[ 45.38072973,  49.04638127,  55.88726129,  63.09275902,  55.095667],
                            [ 47.72076292, 54.28323457,  64.6357677 ,  75.56250393,  65.6040441],
                            [ 51.04456365, 59.73185191,  76.60117439,  91.94453499,  77.09678821],
                            [ 55.0954287 , 68.96095848,  85.03611302, 100.        ,  84.66099388],
                            [ 56.5400143 , 64.88270418,  76.4777017 ,  91.06583877,  76.43104753]])
    
    policy1 = np.array([[3., 3., 2., 2., 2.],
                        [3., 3., 2., 2., 2.],
                        [3., 3., 3., 2., 1.],
                        [3., 3., 3., 4., 1.],
                        [3., 3., 3., 0., 1.]])
    
    value_func2 = np.array([[27.49139846, 27.61387224, 30.43567214, 33.90383366, 39.10661801, 47.63509689, 55.57925131, 62.10159787, 51.92837905, 48.34456612],
                            [24.80202735, 26.08736507, 31.26432514, 37.85961967, 45.92437632, 55.12959504, 65.50263643, 75.53650268, 64.61402631, 55.20567669],
                            [25.38748504, 27.37004179, 32.93138784, 40.75182776, 53.63686929, 65.31139975, 77.8284967 , 92.02982688, 77.03269301, 64.40862473],
                            [26.68224653, 30.40988324, 37.01768797, 45.15666384, 56.23501113, 74.9290622 ,91.96414491 ,99.99999996,  84.25592521, 68.48548764],
                            [30.51794571, 32.69807447, 37.15126075, 42.64867992, 50.50498662, 64.27489226, 76.76437381, 89.77652615, 69.57082465, 58.63162194]])
    
    policy2 = np.array([[3., 3., 3., 3., 3., 3., 2., 2., 2., 2.],
                        [0., 3., 3., 3., 3., 3., 2., 2., 2., 1.],
                        [3., 3., 3., 3., 3., 3., 3., 2., 1., 1.],
                        [2., 3., 3., 3., 3., 3., 3., 4., 1., 1.],
                        [3., 3., 3., 3., 3., 3., 0., 0., 1., 1.]])

    expected_results[0] = [(value_func1, policy1)]
    expected_results[1] = [(value_func2, policy2)]
    
    return expected_results
