from itertools import product
from random import sample, seed

import numpy as np
from pdm4ar.exercises.ex04.structures import Cell
from pdm4ar.exercises_def.ex04.utils import cell2color


def generate_map(shape: tuple[int, int], swamp_percentage: float, n_cliff: int, n_seed) -> np.ndarray:
    # map dimensions should be at least 5x5
    assert shape[0] >= 5 and shape[1] >= 5, "Map dimensions should be at least 5x5"

    seed(n_seed)
    xs, ys = range(0, shape[0]), range(0, shape[1])
    grid_map = Cell.GRASS * np.ones(shape, dtype=int)
    xxyy = list(product(xs, ys))

    assert 0 <= swamp_percentage <= 1, "Swamp percentage should be between 0 and 1"
    # Two cells reserved for goal and start
    swamp_size: int = min(int(swamp_percentage * shape[0] * shape[1]), shape[0] * shape[1] - 2)
    sampled_cells = sample(xxyy, k=swamp_size + 2)

    rows, cols = list(zip(*sampled_cells[2:]))
    grid_map[rows, cols] = Cell.SWAMP

    start_coords = sampled_cells[0]
    # Clip start coords to be inside the map, not near the border
    start_coords = (max(2, min(start_coords[0], shape[0] - 3)), max(2, min(start_coords[1], shape[1] - 3)))
    grid_map[start_coords] = Cell.START
    # Neighbouring cells of start cell are grass
    grid_map[start_coords[0] - 1, start_coords[1]] = Cell.GRASS
    grid_map[start_coords[0] + 1, start_coords[1]] = Cell.GRASS
    grid_map[start_coords[0], start_coords[1] - 1] = Cell.GRASS
    grid_map[start_coords[0], start_coords[1] + 1] = Cell.GRASS

    goal_coords = sampled_cells[1]
    # Move goal if it coincides with start
    if goal_coords == start_coords:
        goal_coords = (goal_coords[0] + 1, goal_coords[1])
    grid_map[goal_coords] = Cell.GOAL

    # Select n_cliff of cliffs from all grass & swamp cells except the 3 by 3 grid centered at the start
    # and four cells exactly 2 cells away from the start.
    grass_swamp_cells = np.where((grid_map == Cell.GRASS) | (grid_map == Cell.SWAMP))
    start_row, start_col = start_coords
    excluded_rows = range(start_row - 1, start_row + 2)
    excluded_cols = range(start_col - 1, start_col + 2)
    excluded_coords = list(product(excluded_rows, excluded_cols))
    excluded_coords.extend(
        [
            (start_row - 2, start_col),
            (start_row + 2, start_col),
            (start_row, start_col - 2),
            (start_row, start_col + 2),
        ]
    )
    available_grass_swamp_coords = [
        (row, col) for row, col in zip(grass_swamp_cells[0], grass_swamp_cells[1]) if (row, col) not in excluded_coords
    ]
    assert len(available_grass_swamp_coords) >= n_cliff, "Not enough grass/swamp cells to place cliffs"

    cliff_coords = sample(available_grass_swamp_coords, k=n_cliff)

    # Place cliffs on the map
    for coord in cliff_coords:
        grid_map[coord] = Cell.CLIFF

    return grid_map


def is_too_close_to_rift_or_border(coord: tuple[int, int], grid_map: np.ndarray) -> bool:
    row, col = coord
    # Check if the cell is within the border
    if row < 2 or row >= grid_map.shape[0] - 2 or col < 2 or col >= grid_map.shape[1] - 2:
        return True
    # Check if the cell is near the cliff
    if grid_map[row - 1 : row + 2, col - 1 : col + 2].max() == Cell.CLIFF:
        return True
    return False


def map2image(map: np.ndarray) -> np.ndarray:
    shape = (*map.shape, 3)
    image = np.zeros(shape)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            image[i, j, :] = cell2color[map[i, j]]
    return image
