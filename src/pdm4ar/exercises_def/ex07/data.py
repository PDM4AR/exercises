from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np

from .structures import *


@dataclass(frozen=True)
class CaseVoyage(IntEnum):
    test_voyage = 0
    random_voyage = 1

    def __eq__(self, other: IntEnum) -> bool:
        return self.value == other.value


def island_generator(
    id: int,
    arch: int,
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    departure_range: Tuple[int, int],
    arrival_range: Tuple[int, int],
    time_compass_range: Tuple[int, int],
    crew_range: Tuple[int, int],
) -> Island:

    x = (x_range[1] - x_range[0]) * np.random.random_sample() + x_range[0]
    y = (y_range[1] - y_range[0]) * np.random.random_sample() + y_range[0]
    departure = (
        departure_range[1] - departure_range[0]
    ) * np.random.random_sample() + departure_range[0]
    arrival = (
        arrival_range[1] - arrival_range[0]
    ) * np.random.random_sample() + arrival_range[0]
    nights = np.random.randint(time_compass_range[0], time_compass_range[1] + 1)
    crew = np.random.randint(crew_range[0], crew_range[1] + 1)

    return Island(id, arch, x, y, departure, arrival, nights, crew)


def island_is_overlapping(
    generated_island: Island, islands: List[Island], viz_radius: float
) -> bool:

    if len(islands) == 0:
        return False
    generated_island_pos = np.array([generated_island.x, generated_island.y]).reshape(
        1, 2
    )
    islands_pos = np.array([[island.x, island.y] for island in islands]).reshape(-1, 2)
    deltas = islands_pos - generated_island_pos
    dist = np.einsum("ij,ij->i", deltas, deltas)
    if np.amin(dist) < (2 * viz_radius) ** 2:
        overlapping = True
    else:
        overlapping = False

    return overlapping


def randomize_reset_constraints(
    constraints: Constraints, p_constraints: Constraints, reset_values: Constraints
) -> Constraints:

    output_constraints = []
    for name_constraint in Constraints.__annotations__.keys():
        if p_constraints is None:
            p_constraints = 1
        p_constraint = max(min(getattr(p_constraints, name_constraint), 1), 0)
        x = np.random.choice(
            [
                getattr(constraints, name_constraint),
                getattr(reset_values, name_constraint),
            ],
            p=[p_constraint, 1 - p_constraint],
        )
        output_constraints.append(x)

    return Constraints(*output_constraints)


def milp_generator(
    milp_seed: int,
    optimization_cost: Optional[IntEnum] = None,
    p_constraints: Optional[List[float]] = None,
) -> ProblemVoyage:

    if optimization_cost is None:
        optimization_costs_enum_values = [cost.value for cost in OptimizationCost]
        optimization_cost = OptimizationCost(
            np.random.choice(optimization_costs_enum_values, 1)
        )

    np.random.seed(milp_seed)
    start_crew = np.random.randint(40, 200)
    n_arch = np.random.randint(4, 25)
    n_islands_arch = np.random.randint(3, 30)
    x_range = (np.random.randint(10, 41), np.random.randint(80, 120))
    y_range = (np.random.randint(10, 30), np.random.randint(110, 150))
    departure_range = (np.random.randint(5, 8), np.random.randint(9, 13))
    arrival_range = (np.random.randint(15, 18), np.random.randint(19, 21))
    time_compass_range = (np.random.randint(1, 5), np.random.randint(6, 9))
    crew_range = (np.random.randint(-19, 0), np.random.randint(3, 20))
    x_offset_levels = np.random.randint(
        x_range[1] - x_range[0] + 0, x_range[1] - x_range[0] + 10
    )
    y_offset_levels = [-10, 10]

    max_distance_individual_sail = (
        0.8 * (y_range[1] - y_range[0])
        + 0.8 * (x_range[1] - x_range[0])
        + x_offset_levels
    )
    min_theoretical_sail = arrival_range[0] - departure_range[1]
    max_theoretical_sail = arrival_range[1] - departure_range[0]
    max_duration_individual_sail = min_theoretical_sail + 0.7 * (
        max_theoretical_sail - min_theoretical_sail
    )
    min_fix_ship_individual_island = round(
        0.5 * time_compass_range[0]
        + np.random.random() * (time_compass_range[1] - time_compass_range[0])
    )
    mean_crew_change = (crew_range[1] - crew_range[0]) / 2
    if mean_crew_change > 0:
        max_total_crew = start_crew + int(1.8 * n_arch * (mean_crew_change))
        min_total_crew = max(round(0.8 * start_crew), 1)
    elif mean_crew_change < 0:
        max_total_crew = round(1.2 * start_crew)
        min_total_crew = max(start_crew - int(1.8 * n_arch * (mean_crew_change)), 1)
    else:
        max_total_crew = round(1.8 * start_crew)
        min_total_crew = max(round(0.6 * start_crew), 1)

    islands = []
    max_trials_samples = 200
    island_radius_viz = (
        2  # THIS IS JUST FOR VISUALIZATION, NO RELATED TO PROBLEM SOLVING
    )
    island_id = 0
    for k in range(n_arch):
        x_range = [i + x_offset_levels for i in x_range]
        y_offset = np.random.randint(y_offset_levels[0], y_offset_levels[1] + 1)
        y_range = [i + y_offset for i in y_range]
        time_compass_range_tmp = [0, 0] if k in (0, n_arch - 1) else time_compass_range
        crew_range_tmp = [0, 0] if k in (0, n_arch - 1) else crew_range
        n_islands_arch_tmp = 1 if k in (0, n_arch - 1) else n_islands_arch
        for _ in range(n_islands_arch_tmp):
            pos_ok = False
            n_trials_samples = 0
            while pos_ok is not True:
                if n_trials_samples >= max_trials_samples:
                    raise ValueError(
                        f"Exceeded the amount of trials ({n_trials_samples}) to generate a non overlapping island."
                        f" Try to increase the x_range ({x_range}) or y_range ({y_range}), or run again with a different seed."
                    )
                generated_island = island_generator(
                    island_id,
                    k,
                    x_range,
                    y_range,
                    departure_range,
                    arrival_range,
                    time_compass_range_tmp,
                    crew_range_tmp,
                )
                pos_ok = not island_is_overlapping(
                    generated_island, islands, island_radius_viz
                )
                n_trials_samples += 1
            islands.append(generated_island)
            island_id += 1
    islands = tuple(islands)

    constraints = Constraints(
        min_fix_ship_individual_island,
        min_total_crew,
        max_total_crew,
        max_duration_individual_sail,
        max_distance_individual_sail,
    )
    if p_constraints is None:
        p_constraints = Constraints(*[0.5 for _ in constraints])
    reset_values = Constraints(None, 1, None, None, None)
    constraints = randomize_reset_constraints(constraints, p_constraints, reset_values)

    problem = ProblemVoyage(optimization_cost, start_crew, islands, constraints)

    return problem
