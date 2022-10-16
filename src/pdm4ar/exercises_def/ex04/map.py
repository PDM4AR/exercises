from itertools import product
from random import sample, seed
from typing import Tuple

import numpy as np

from pdm4ar.exercises.ex04.structures import Cell
from pdm4ar.exercises_def.ex04.utils import cell2color


def generate_map(shape: Tuple[int, int], swamp_percentage: float, n_seed=3) -> np.ndarray:
    seed(n_seed)
    xs, ys = range(0, shape[0]), range(0, shape[1])
    map = Cell.GRASS * np.ones(shape, dtype=int)
    xxyy = list(product(xs, ys))

    assert 0 <= swamp_percentage <= 1
    # Two cells reserved for goal and start
    swamp_size: int = min(int(swamp_percentage * shape[0] * shape[1]), shape[0] * shape[1] - 2)
    sampled_cells = sample(xxyy, k=swamp_size + 2)

    map[sampled_cells[0]] = Cell.START
    map[sampled_cells[1]] = Cell.GOAL
    rows, cols = list(zip(*sampled_cells[2:]))
    map[rows, cols] = Cell.SWAMP
    return map


def map2image(map: np.ndarray) -> np.ndarray:
    shape = (*map.shape, 3)
    image = np.zeros(shape)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            image[i, j, :] = cell2color[map[i, j]]
    return image
