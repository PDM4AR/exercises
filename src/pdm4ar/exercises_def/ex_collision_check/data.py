import math
import random
from typing import List, Tuple

import numpy as np
from geometry.poses import SE2_from_translation_angle
from shapely import geometry

from .structures import *


class DataGenerator:
    MAP_INFO = [
        {
            "obstacles": [
                {
                    "center": (150, 150),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
                {
                    "center": (50, 100),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                    "num_vertices": 5,
                },
                {
                    "center": (45, 45),
                    "avg_radius": 15,
                    "irregularity": 0,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
            ],
            "path": [
                (0, 0),
                (30, 45),
                (60, 60),
                (90, 75),
                (120, 110),
                (130, 155),
            ],
        },
        {
            "obstacles": [
                {
                    "center": (150, 150),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
                {
                    "center": (50, 100),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 5,
                },
                {
                    "center": (45, 45),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 4,
                },
            ],
            "path": [
                (0, 0),
                (30, 45),
                (60, 60),
                (90, 75),
                (120, 110),
                (130, 155),
            ],
        },
        {
            "obstacles": [
                {
                    "center": (125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (25, 35),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
                {
                    "center": (80, 45),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.2,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            ],
            "path": [
                (0, 0),
                (10, 15),
                (35, 45),
                (60, 55),
                (80, 80),
                (100, 95),
                (120, 115),
                (130, 155),
            ],
        },
        {
            "obstacles": [
                {
                    "center": (125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 8,
                },
                {
                    "center": (25, 35),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
                {
                    "center": (80, 45),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.6,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
            ],
            "path": [
                (0, 0),
                (10, 15),
                (35, 45),
                (60, 55),
                (80, 80),
                (100, 95),
                (120, 115),
                (130, 155),
            ],
        },
        {
            "obstacles": [
                {
                    "center": (125, 125),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 8,
                },
                {
                    "center": (25, 35),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
                {
                    "center": (80, 45),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 6,
                },
                {
                    "center": (0, 100),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 9,
                },
                {
                    "center": (100, 0),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 7,
                },
                {
                    "center": (100, 80),
                    "avg_radius": 15,
                    "irregularity": 0.8,
                    "spikiness": 0,
                    "num_vertices": 8,
                },
            ],
            "path": [
                (0, 0),
                (10, 15),
                (35, 45),
                (60, 55),
                (80, 80),
                (100, 95),
                (120, 115),
                (130, 155),
            ],
        },
    ]

    @staticmethod
    def generate_random_point(
        min_dist: float, max_dist: float, center: Point = Point(0, 0)
    ) -> Point:
        # Distance of point to given center
        dist = np.random.uniform(min_dist, max_dist)
        # Angle in Polar Coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        # Generate New Point
        return Point(
            center.x + dist * np.cos(theta),
            center.y + dist * np.sin(theta),
        )

    @staticmethod
    def generate_random_circle() -> Circle:
        # Generate Center of the Circle
        center = DataGenerator.generate_random_point(0, 20)
        # Generate Random Radius
        radius = np.random.uniform(3, 10)

        return Circle(center, radius)

    @staticmethod
    def generate_random_triangle() -> Triangle:
        # Generate 3 Points Randomly
        p1 = DataGenerator.generate_random_point(0, 20)
        p2 = DataGenerator.generate_random_point(0, 20)
        p3 = DataGenerator.generate_random_point(0, 20)

        return Triangle(p1, p2, p3)

    @staticmethod
    def generate_polygon(
        center: Tuple[float, float],
        avg_radius: float,
        irregularity: float,
        spikiness: float,
        num_vertices: int,
    ) -> Polygon:
        """
        This code is taken from here => https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

        Start with the center of the polygon at center, then creates the
        polygon by sampling points on a circle around the center.
        Random noise is added by varying the angular spacing between
        sequential points, and by varying the radial distance of each
        point from the centre.
        Args:
            center (Tuple[float, float]):
                a pair representing the center of the circumference used
                to generate the polygon.
            avg_radius (float):
                the average radius (distance of each generated vertex to
                the center of the circumference) used to generate points
                with a normal distribution.
            irregularity (float):
                variance of the spacing of the angles between consecutive
                vertices.
            spikiness (float):
                variance of the distance of each vertex to the center of
                the circumference.
            num_vertices (int):
                the number of vertices of the polygon.
        Returns:
            Polygon: polygon, in CCW order.
        """
        # Parameter check
        if irregularity < 0 or irregularity > 1:
            raise ValueError("Irregularity must be between 0 and 1.")
        if spikiness < 0 or spikiness > 1:
            raise ValueError("Spikiness must be between 0 and 1.")

        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = DataGenerator.random_angle_steps(
            num_vertices, irregularity
        )

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = np.clip(
                random.gauss(avg_radius, spikiness), 0, 2 * avg_radius
            )
            point = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
            points.append(point)
            angle += angle_steps[i]

        return points

    @staticmethod
    def random_angle_steps(steps: int, irregularity: float) -> List[float]:
        """
        This code is taken from here => https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

        Generates the division of a circumference in random angles.
        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            List[float]: the list of the random angles.
        """
        # generate n angle steps
        angles = []
        lower = (2 * math.pi / steps) - irregularity
        upper = (2 * math.pi / steps) + irregularity
        cumsum = 0
        for i in range(steps):
            angle = random.uniform(lower, upper)
            angles.append(angle)
            cumsum += angle

        # normalize the steps so that point 0 and point n+1 are the same
        cumsum /= 2 * math.pi
        for i in range(steps):
            angles[i] /= cumsum
        return angles

    @staticmethod
    def generate_random_polygon(
        center: Point = Point(0, 0),
        avg_radius: float = 3.0,
        irregularity: float = 0.5,
        spikiness: float = 0,
        num_vertices: int = (4, 9),
    ) -> Polygon:
        # Randomly Select Number of Corners
        if type(num_vertices) is tuple:
            num_vertices = np.random.randint(*num_vertices)

        poly_points = DataGenerator.generate_polygon(
            (center.x, center.y),
            avg_radius,
            irregularity,
            spikiness,
            num_vertices,
        )

        return Polygon([Point(x, y) for (x, y) in poly_points])

    @staticmethod
    def generate_circle_point_collision_data(
        index: int,
    ) -> Tuple[Circle, Point, bool]:
        # Generate Random Circle
        circle = DataGenerator.generate_random_circle()
        # Generate Point
        if np.random.uniform() < 0.5:
            point = DataGenerator.generate_random_point(
                0, circle.radius, circle.center
            )
            return (circle, point, True)
        else:
            point = DataGenerator.generate_random_point(
                circle.radius, 2 * circle.radius, circle.center
            )
            return (circle, point, False)

    @staticmethod
    def generate_triangle_point_collision_data(
        index: int,
    ) -> Tuple[Triangle, Point, bool]:
        # Generate Random Triangle
        triangle = DataGenerator.generate_random_triangle()
        # Calculate Center of the Triangle
        center = triangle.center()
        # Generate Query Point
        point = DataGenerator.generate_random_point(1, 5, center)
        # Check Collision via Shapely
        triangle_shapely = geometry.Polygon(
            [
                [triangle.v1.x, triangle.v1.y],
                [triangle.v2.x, triangle.v2.y],
                [triangle.v3.x, triangle.v3.y],
            ]
        )
        point_shapely = geometry.Point(point.x, point.y)

        return (
            triangle,
            point,
            triangle_shapely.distance(point_shapely) < 1e-5,
        )

    @staticmethod
    def generate_polygon_point_collision_data(
        index: int,
    ) -> Tuple[Polygon, Point, bool]:
        # Generate Random Polygon
        poly = DataGenerator.generate_random_polygon()
        # Calculate Center of the Corners
        center = poly.center()
        # Generate Query Point
        point = DataGenerator.generate_random_point(1, 5, center)
        # Check Collision via Shapely
        poly_shapely = geometry.Polygon(
            [[p.x, p.y] for p in poly.vertices]
        )
        point_shapely = geometry.Point(point.x, point.y)

        return (poly, point, poly_shapely.distance(point_shapely) < 1e-5)

    @staticmethod
    def generate_circle_line_collision_data(
        index: int,
    ) -> Tuple[Circle, Line, bool]:
        # Generate Random Circle
        circle = DataGenerator.generate_random_circle()
        # Generate Points
        p1 = DataGenerator.generate_random_point(
            0, 2 * circle.radius, circle.center
        )
        p2 = DataGenerator.generate_random_point(
            0, 2 * circle.radius, circle.center
        )
        # Check Collision via Shapely
        circle_shapely = geometry.Point(
            circle.center.x, circle.center.y
        ).buffer(circle.radius)
        line_shapely = geometry.LineString(
            [geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)]
        )

        return (
            circle,
            Line(p1, p2),
            circle_shapely.distance(line_shapely) < 1e-5,
        )

    @staticmethod
    def generate_polygon_line_collision_data(
        index: int,
    ) -> Tuple[Polygon, Line, bool]:
        # Generate Random Polygon
        poly = DataGenerator.generate_random_polygon()
        # Calculate Center of the Corners
        center = poly.center()
        # Calculate Max Distance to Corners
        max_dist = max(
            [
                ((center.x - c.x) ** 2 + (center.y - c.y) ** 2) ** 0.5
                for c in poly.vertices
            ]
        )
        # Generate Points
        p1 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        p2 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        # Check Collisions via Shapely
        poly_shapely = geometry.Polygon(
            [[v.x, v.y] for v in poly.vertices]
        )
        line_shapely = geometry.LineString(
            [geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)]
        )

        return (
            poly,
            Line(p1, p2),
            poly_shapely.distance(line_shapely) < 1e-5,
        )

    @staticmethod
    def generate_random_robot_map_and_path(
        index: int,
    ) -> Tuple[Path, float, List[Polygon], List[int]]:

        # Generate Random Robot Radius
        r = float(np.random.randint(3, 7))

        # Generate Path
        path = Path(
            [
                Point(x, y)
                for (x, y) in DataGenerator.MAP_INFO[
                    index % len(DataGenerator.MAP_INFO)
                ]["path"]
            ]
        )
        # Generate Obstacles
        obstacles = [
            DataGenerator.generate_random_polygon(
                Point(poly["center"][0], poly["center"][1]),
                poly["avg_radius"],
                poly["irregularity"],
                poly["spikiness"],
                poly["num_vertices"],
            )
            for poly in DataGenerator.MAP_INFO[
                index % len(DataGenerator.MAP_INFO)
            ]["obstacles"]
        ]

        # Check collision for ground truth
        ground_truth = []
        # Convert obstacles to Shapely Shapes
        shapely_obstacles = [
            geometry.Polygon([[p.x, p.y] for p in poly.vertices])
            for poly in obstacles
        ]
        # Check distance between each line segment with each polygon
        for i, (p1, p2) in enumerate(
            zip(path.waypoints[:-1], path.waypoints[1:])
        ):
            ls_shapely = geometry.LineString(
                [geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)]
            )
            for obs in shapely_obstacles:
                if ls_shapely.distance(obs) < r:
                    ground_truth.append(i)
                    break

        return (path, r, obstacles, ground_truth)

    @staticmethod
    def generate_robot_frame_data(
        index: int,
    ) -> Tuple[
        List[Pose2D], float, List[List[Polygon]], List[Polygon], List[int]
    ]:
        # Initialize Random Map
        (
            path,
            r,
            obstacles,
            ground_truth,
        ) = DataGenerator.generate_random_robot_map_and_path(index)
        # Calculate Robot Poses
        # In every waypoint robot will turn into its next position
        poses = []
        for wp_1, wp_2 in zip(path.waypoints[:-1], path.waypoints[1:]):
            wp_temp = Point(wp_2.x - wp_1.x, wp_2.y - wp_1.y)
            theta = np.arctan2(wp_temp.y, wp_temp.x)

            poses.append(Pose2D(wp_1, theta))
        # Append Last Pose with the Latest Theta
        poses.append(Pose2D(path.waypoints[-1], theta))
        # Calculate Observed Obstacles
        observation_radius = 50
        shapely_obstacles = [
            geometry.Polygon([[p.x, p.y] for p in poly.vertices])
            for poly in obstacles
        ]
        observations = []
        for pose in poses:
            observations.append([])
            # Check distance to obstacles
            shapely_point = geometry.Point(
                pose.position.x, pose.position.y
            )
            for shapely_obs, obs in zip(shapely_obstacles, obstacles):
                if (
                    shapely_point.distance(shapely_obs)
                    < observation_radius + r
                ):
                    # Calculate position of the obstacle in robot frame
                    robot_frame_poly = obs.apply_SE2transform(
                        SE2_from_translation_angle(
                            -np.array([pose.position.x, pose.position.y]),
                            0,
                        )
                    ).apply_SE2transform(
                        SE2_from_translation_angle(
                            np.array([0, 0]),
                            -pose.theta,
                        )
                    )
                    observations[-1].append(robot_frame_poly)

        return (poses, r, observations, obstacles, ground_truth)
