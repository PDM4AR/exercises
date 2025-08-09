from typing import List, Tuple

import cvxpy as cp
import numpy as np
from pdm4ar.exercises_def.ex06.structures import (
    Capsule,
    Circle,
    GeoPrimitive,
    Polygon,
    Triangle,
)


class OptCollisionCheckingPrimitives:
    """
    Optimization-based collision detection for geometric primitives.
    """

    @staticmethod
    def check_collision(obs1: GeoPrimitive, obs2: GeoPrimitive) -> Tuple[bool, float]:
        """
        Check collision between two geometric primitives using convex optimization.

        Args:
            obs1 (GeoPrimitive): First geometric primitive (Triangle, Polygon, Circle, or Capsule)
            obs2 (GeoPrimitive): Second geometric primitive (Triangle, Polygon, Circle, or Capsule)

        Returns:
            Tuple[bool, float]: Tuple containing:
                - collision_exists (bool): True if primitives are in collision
                - minimum_distance (float): Scaling factor at contact (<=1.0 indicates collision)
        """
        # Optimization variables
        x = cp.Variable(2, name="x")  # Contact point in 2D space
        scale = cp.Variable(1, name="scale")  # Scaling factor for primitives

        # TODO: Construct collision constraints based on primitive types and solve the optimization problem

        # Placeholder return value: returns True (have collision) and 1.0 (default scaling factor)
        # The scaling factor of 1.0 here means two primitives are just touching at their boundaries.
        # You need to implement the actual collision checking logic.
        return True, 1.0

    @staticmethod
    def _add_collision_constraints_polygon(polygon: Polygon, x: cp.Variable, scale: cp.Variable) -> List[cp.Constraint]:
        """
        Generate collision constraints for a polygon primitive.

        Creates half-space constraints for each edge of the polygon. The point x
        must satisfy all constraints to be inside the scaled polygon.

        Args:
            polygon (Polygon): Polygon primitive with vertices
            x (cp.Variable): Optimization variable representing the contact point. Dimension: (2,)
            scale (cp.Variable): Optimization variable for scaling factor. Dimension: (1,)

        Returns:
            List[cp.Constraint]: List of linear constraints ensuring x is inside the scaled polygon
        """

        # TODO: Implement the constraints for the polygon

        constraints = []
        return constraints

    @staticmethod
    def _add_collision_constraints_circle(circle: Circle, x: cp.Variable, scale: cp.Variable) -> List[cp.Constraint]:
        """
        Generate collision constraints for a circular primitive.

        Creates a constraint ensuring the contact point x is within the scaled circle.

        Args:
            circle (Circle): Circle primitive with center and radius
            x (cp.Variable): Optimization variable representing the contact point. Dimension: (2,)
            scale (cp.Variable): Optimization variable for scaling factor. Dimension: (1,)

        Returns:
            List[cp.Constraint]: List containing single constraint for circular boundary
        """
        return [cp.norm(x - np.array([circle.center.x, circle.center.y])) <= circle.radius * scale]

    @staticmethod
    def _add_collision_constraints_capsule(capsule: Capsule, x: cp.Variable, scale: cp.Variable) -> List[cp.Constraint]:
        """
        Add collision constraints for a capsule obstacle.

        The capsule is represented as a polygon with two circular ends.

        You can model the capsule by constraining the euclidean distance from the point x to a line segment is less than or equal to the radius of the capsule.

        Args:
            capsule (Capsule): The capsule obstacle.
            x (cp.Variable): The variable representing the point to check. Dimension: (2,)
            scale (cp.Variable): The scaling factor for the capsule. Dimension: (1,)

        Returns:
            List[cp.Constraint]: A list of constraints that ensure the point x does not penetrate the capsule.
        """

        # TODO: Implement the constraints for the capsule

        constraints = []
        return constraints

    @staticmethod
    def _add_collision_constraints_triangle(
        triangle: Triangle, x: cp.Variable, scale: cp.Variable
    ) -> List[cp.Constraint]:
        """
        Generate collision constraints for a triangular primitive.

        Converts the triangle to a polygon and uses polygon constraints.

        Args:
            triangle (Triangle): Triangle primitive with three vertices
            x (cp.Variable): Optimization variable representing the contact point. Dimension: (2,)
            scale (cp.Variable): Optimization variable for scaling factor. Dimension: (1,)

        Returns:
            List[cp.Constraint]: List of constraints ensuring x is inside the scaled triangle
        """
        # Convert triangle to polygon and use polygon constraints
        return OptCollisionCheckingPrimitives._add_collision_constraints_polygon(
            Polygon([triangle.v1, triangle.v2, triangle.v3]), x, scale
        )
