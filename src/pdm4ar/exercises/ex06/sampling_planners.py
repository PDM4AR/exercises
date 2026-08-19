"""Sampling-based motion-planning interfaces for Exercise 06.

The planners use the collision checker implemented in Tasks 1--5. PRM is
given a deterministic set of samples, while RRT* owns its sampling strategy.
"""

from pdm4ar.exercises_def.ex06.structures import AABB, GeoPrimitive, Path, Point


class SamplingBasedPlanner:
    """Student API for the planning applications in Tasks 6 and 7."""

    @staticmethod
    def prm(
        samples: list[Point],
        queries: list[tuple[Point, Point]],
        bounds: AABB,
        robot_radius: float,
        obstacles: list[GeoPrimitive],
        connection_radius: float,
    ) -> list[Path]:
        """Build a collision-free roadmap and solve every start--goal query.

        ``samples`` is generated deterministically by the evaluator and may
        contain configurations that collide with obstacles. Every returned
        path must use only valid configurations from ``samples``. Return an
        empty ``Path`` for a query that cannot be solved.
        """
        # TODO: Task 6
        raise NotImplementedError

    @staticmethod
    def rrt_star(
        start: Point,
        goal: Point,
        bounds: AABB,
        robot_radius: float,
        obstacles: list[GeoPrimitive],
        max_iterations: int = 3000,
        step_size: float = 0.5,
        rewire_radius: float = 1.5,
        goal_bias: float = 0.1,
        seed: int = 0,
    ) -> Path:
        """Plan a path with RRT*, including parent selection and rewiring.

        Sampling may be implemented in any way, but using ``seed`` must make a
        run reproducible. Return ``Path([])`` if no solution is found.
        """
        # TODO: Task 7
        raise NotImplementedError
