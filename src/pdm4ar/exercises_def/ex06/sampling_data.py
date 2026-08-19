"""Deterministic public scenarios for the planning tasks in Exercise 06."""

from __future__ import annotations

import numpy as np

from pdm4ar.exercises_def.ex06.structures import (
    AABB,
    Circle,
    GeoPrimitive,
    Point,
    Polygon,
    Triangle,
)


PUBLIC_PRM_CASES = 5
PUBLIC_RRT_STAR_CASES = 10


def _rectangle(x_min: float, y_min: float, x_max: float, y_max: float) -> Polygon:
    return Polygon(
        [
            Point(x_min, y_min),
            Point(x_max, y_min),
            Point(x_max, y_max),
            Point(x_min, y_max),
        ]
    )


class SamplingDataGenerator:
    """Generate reproducible PRM sample sets and RRT* problems."""

    _PRM_REFERENCE_COSTS: tuple[tuple[float, ...], ...] = (
        (12.256080917071122, 12.224157494128782, 8.162080166701468),
        (13.537077194235485, 13.354314076816229, 10.10036040366784),
        (13.484512999853473, 12.216539440221734, 8.023340960574469),
        (12.915231432887742, 13.228416092222586, 10.139266922945055),
        (13.523425546242116, 12.8877485157809, 9.832088850140536),
    )
    _RRT_STAR_REFERENCE_COSTS: tuple[float, ...] = (
        12.16223663640862,
        13.054006917166516,
        13.109045252083178,
        12.469072933389455,
        13.273172182386427,
        13.294138755093575,
        12.605472628868762,
        12.766118546675115,
        12.187332875660617,
        12.271381262649564,
    )

    @staticmethod
    def _environment(
        index: int,
    ) -> tuple[Point, Point, AABB, float, list[GeoPrimitive]]:
        bounds = AABB(Point(0.0, 0.0), Point(10.0, 10.0))
        start, goal = Point(0.7, 0.7), Point(9.3, 9.3)
        scenarios: list[list[GeoPrimitive]] = [
            [],
            [_rectangle(4.0, 3.0, 6.0, 7.0)],
            [_rectangle(2.0, 2.0, 4.5, 4.0), _rectangle(5.5, 6.0, 8.0, 8.0)],
            [
                Circle(Point(8.5, 2.0), 0.45),
                Triangle(Point(3.8, 2.5), Point(6.2, 5.0), Point(3.8, 7.5)),
                _rectangle(1.0, 7.5, 2.0, 9.0),
            ],
            [_rectangle(3.8, 0.0, 5.0, 7.2)],
            [_rectangle(2.0, 4.0, 8.0, 5.2)],
            [
                _rectangle(1.8, 2.0, 5.8, 3.0),
                _rectangle(4.2, 5.0, 8.2, 6.0),
            ],
            [
                Circle(Point(2.6, 3.5), 0.8),
                Circle(Point(5.0, 5.0), 0.9),
                Circle(Point(7.4, 6.5), 0.8),
            ],
            [_rectangle(4.2, 0.0, 5.8, 4.1), _rectangle(4.2, 5.9, 5.8, 10.0)],
            [_rectangle(0.0, 4.2, 3.5, 5.8), _rectangle(5.1, 4.2, 10.0, 5.8)],
        ]
        return start, goal, bounds, 0.2, scenarios[index % len(scenarios)]

    @staticmethod
    def _samples(
        bounds: AABB,
        seed: int,
        count: int,
        required_points: list[Point],
    ) -> list[Point]:
        """Return fixed raw samples; collision filtering belongs to Task 6."""
        rng = np.random.default_rng(seed)
        random_samples = [
            Point(
                float(rng.uniform(bounds.p_min.x, bounds.p_max.x)),
                float(rng.uniform(bounds.p_min.y, bounds.p_max.y)),
            )
            for _ in range(count)
        ]
        return required_points + random_samples

    @staticmethod
    def generate_prm(index: int):
        if not 0 <= index < PUBLIC_PRM_CASES:
            raise IndexError(f"public PRM case must be in [0, {PUBLIC_PRM_CASES})")
        start, goal, bounds, radius, obstacles = SamplingDataGenerator._environment(
            index
        )
        queries = [
            (start, goal),
            (Point(0.7, 9.3), Point(9.3, 0.7)),
            (Point(1.0, 5.0), Point(9.0, 5.0)),
        ]
        required = [point for query in queries for point in query]
        samples = SamplingDataGenerator._samples(bounds, 20_000 + index, 320, required)
        expected = (
            SamplingDataGenerator._PRM_REFERENCE_COSTS[index]
            if SamplingDataGenerator._PRM_REFERENCE_COSTS
            else ()
        )
        return samples, queries, bounds, radius, obstacles, 1.7, expected

    @staticmethod
    def generate_rrt_star(index: int):
        if not 0 <= index < PUBLIC_RRT_STAR_CASES:
            raise IndexError(
                f"public RRT* case must be in [0, {PUBLIC_RRT_STAR_CASES})"
            )
        start, goal, bounds, radius, obstacles = SamplingDataGenerator._environment(
            index
        )
        reference_cost = (
            SamplingDataGenerator._RRT_STAR_REFERENCE_COSTS[index]
            if SamplingDataGenerator._RRT_STAR_REFERENCE_COSTS
            else float("nan")
        )
        return (
            start,
            goal,
            bounds,
            radius,
            obstacles,
            5000,
            0.55,
            1.35,
            0.12,
            30_000 + index,
            reference_cost,
        )
