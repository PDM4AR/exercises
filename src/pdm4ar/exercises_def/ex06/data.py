import math
import random
import warnings
from typing import Union

import numpy as np
from dg_commons import SE2Transform
from numpy.linalg import inv
from shapely import geometry

from .map_config import EXERCISE_MAP_CONFIGS
from .structures import Circle, GeoPrimitive, Path, Point, Polygon, Segment, Triangle


class DataGenerator:
    @staticmethod
    def generate_random_point(min_dist: float, max_dist: float, center: Point = Point(0, 0)) -> Point:
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
    def generate_random_circle(
        center: Point = Point(0, 0),
        max_dist: float = 20,
        min_radius: float = 3,
        max_radius: float = 10,
    ) -> Circle:
        # Generate Center of the Circle
        center = DataGenerator.generate_random_point(0, max_dist, center)
        # Generate Random Radius
        radius = np.random.uniform(min_radius, max_radius)

        return Circle(center, radius)

    @staticmethod
    def generate_random_triangle(
        center: Point = Point(0, 0),
        avg_radius: float = 5,
        irregularity: float = 0.5,
        spikiness: float = 0.5,
    ) -> Triangle:
        # Generate 3 Points Randomly
        triangle_vertices = DataGenerator.generate_polygon_vertices(
            (center.x, center.y),
            avg_radius,
            irregularity,
            spikiness,
            3,
        )

        return Triangle(
            Point(*triangle_vertices[0]),
            Point(*triangle_vertices[1]),
            Point(*triangle_vertices[2]),
        )

    @staticmethod
    def generate_polygon_vertices(
        center: tuple[float, float],
        avg_radius: float,
        irregularity: float,
        spikiness: float,
        num_vertices: int,
    ) -> list[tuple[float, float]]:
        """
        This code is taken from here => https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

        Start with the center of the polygon at center, then creates the
        polygon by sampling points on a circle around the center.
        Random noise is added by varying the angular spacing between
        sequential points, and by varying the radial distance of each
        point from the centre.
        Args:
            center (tuple[float, float]):
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
            vertices: the list of points representing the vertices of the polygon.
        """
        # Parameter check
        if irregularity < 0 or irregularity > 1:
            raise ValueError("Irregularity must be between 0 and 1.")
        if spikiness < 0 or spikiness > 1:
            raise ValueError("Spikiness must be between 0 and 1.")

        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = DataGenerator.random_angle_steps(num_vertices, irregularity)

        # now generate the points
        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = np.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
            point = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
            points.append(point)
            angle += angle_steps[i]

        return points

    @staticmethod
    def random_angle_steps(steps: int, irregularity: float) -> list[float]:
        """
        This code is taken from here => https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

        Generates the division of a circumference in random angles.
        Args:
            steps (int):
                the number of angles to generate.
            irregularity (float):
                variance of the spacing of the angles between consecutive vertices.
        Returns:
            list[float]: the list of the random angles.
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
        num_vertices: Union[int, tuple[int, int]] = (4, 9),
    ) -> Polygon:
        # Randomly Select Number of Corners
        if isinstance(num_vertices, tuple):
            num_vertices = np.random.randint(*num_vertices)

        poly_points = DataGenerator.generate_polygon_vertices(
            (center.x, center.y),
            avg_radius,
            irregularity,
            spikiness,
            num_vertices,
        )

        def _is_valid_polygon(polygon: Polygon) -> bool:
            # Check if the polygon is valid (convex, no colliding points, at least 3 vertices, and vertices are in CCW order)
            if len(polygon.vertices) < 3:
                return False
            for i in range(len(polygon.vertices)):  # pylint: disable=consider-using-enumerate
                p1 = polygon.vertices[i]
                p2 = polygon.vertices[(i + 1) % len(polygon.vertices)]
                p3 = polygon.vertices[(i + 2) % len(polygon.vertices)]
                cross_product = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
                if cross_product < 0:
                    return False
            return True

        polygon = Polygon([Point(x, y) for (x, y) in poly_points])

        if not _is_valid_polygon(polygon):
            warnings.warn("Generated polygon is not valid. Please try again.")

        return polygon

    @staticmethod
    def generate_circle_point_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Circle, Point, bool]:
        # Generate Random Circle
        circle = DataGenerator.generate_random_circle()
        # Generate Point
        if np.random.uniform() < 0.5:
            point = DataGenerator.generate_random_point(0, circle.radius, circle.center)
            return (circle, point, True)
        else:
            point = DataGenerator.generate_random_point(circle.radius, 2 * circle.radius, circle.center)
            return (circle, point, False)

    @staticmethod
    def generate_triangle_point_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Triangle, Point, bool]:
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
            triangle_shapely.distance(point_shapely) == 0.0,
        )

    @staticmethod
    def generate_polygon_point_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Polygon, Point, bool]:
        # Generate Random Polygon
        poly = DataGenerator.generate_random_polygon()
        # Calculate Center of the Corners
        center = poly.center()
        # Generate Query Point
        point = DataGenerator.generate_random_point(1, 5, center)
        # Check Collision via Shapely
        poly_shapely = geometry.Polygon([[p.x, p.y] for p in poly.vertices])
        point_shapely = geometry.Point(point.x, point.y)

        return poly, point, poly_shapely.distance(point_shapely) == 0.0

    @staticmethod
    def generate_circle_segment_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Circle, Segment, bool]:
        # Generate Random Circle
        circle = DataGenerator.generate_random_circle()
        # Generate Points
        p1 = DataGenerator.generate_random_point(0, 2 * circle.radius, circle.center)
        p2 = DataGenerator.generate_random_point(0, 2 * circle.radius, circle.center)
        # Check Collision via Shapely
        circle_shapely = geometry.Point(circle.center.x, circle.center.y).buffer(circle.radius)
        segment_shapely = geometry.LineString([geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)])

        return (
            circle,
            Segment(p1, p2),
            circle_shapely.distance(segment_shapely) == 0.0,
        )

    @staticmethod
    def generate_tringle_segment_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Triangle, Segment, bool]:
        # Generate Random Polygon
        triangle = DataGenerator.generate_random_triangle()
        # Calculate Center of the Corners
        center = triangle.center()
        # Calculate Max Distance to Corners
        max_dist = max(
            [
                ((center.x - triangle.v1.x) ** 2 + (center.y - triangle.v1.y) ** 2) ** 0.5,
                ((center.x - triangle.v1.x) ** 2 + (center.y - triangle.v1.y) ** 2) ** 0.5,
                ((center.x - triangle.v1.x) ** 2 + (center.y - triangle.v1.y) ** 2) ** 0.5,
            ]
        )
        # Generate Points
        p1 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        p2 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        # Check Collisions via Shapely
        poly_shapely = geometry.Polygon(
            [
                [triangle.v1.x, triangle.v1.y],
                [triangle.v2.x, triangle.v2.y],
                [triangle.v3.x, triangle.v3.y],
            ]
        )
        segment_shapely = geometry.LineString([geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)])

        return (
            triangle,
            Segment(p1, p2),
            poly_shapely.distance(segment_shapely) == 0.0,
        )

    @staticmethod
    def generate_polygon_segment_collision_data(
        index: int,  # pylint: disable=unused-argument
    ) -> tuple[Polygon, Segment, bool]:
        # Generate Random Polygon
        poly = DataGenerator.generate_random_polygon()
        # Calculate Center of the Corners
        center = poly.center()
        # Calculate Max Distance to Corners
        max_dist = max([((center.x - c.x) ** 2 + (center.y - c.y) ** 2) ** 0.5 for c in poly.vertices])
        # Generate Points
        p1 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        p2 = DataGenerator.generate_random_point(0, 2 * max_dist, center)
        # Check Collisions via Shapely
        poly_shapely = geometry.Polygon([[v.x, v.y] for v in poly.vertices])
        segment_shapely = geometry.LineString([geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)])

        return (
            poly,
            Segment(p1, p2),
            poly_shapely.distance(segment_shapely) == 0.0,
        )

    @staticmethod
    def generate_random_robot_map_and_path(
        exercise_id: int,
        index: int,
    ) -> tuple[Path, float, list[GeoPrimitive], list[int]]:

        map_config = EXERCISE_MAP_CONFIGS[exercise_id]
        # Generate Random Robot Radius
        r = float(np.random.randint(3, 7))

        # Generate Path
        path = Path([Point(x, y) for (x, y) in map_config[index % len(map_config)]["path"]])
        # Generate Obstacles
        obstacles = []
        for obs in map_config[index % len(map_config)]["obstacles"]:
            obs_generation_func = getattr(DataGenerator, f"generate_random_{obs['type']}")
            obstacles.append(obs_generation_func(**obs["params"]))

        # Check collision for ground truth
        ground_truth = []
        # Convert obstacles to Shapely Shapes
        shapely_obstacles = []
        for obs in obstacles:
            if isinstance(obs, Polygon):
                shapely_obstacles.append(geometry.Polygon([[p.x, p.y] for p in obs.vertices]))
            elif isinstance(obs, Triangle):
                shapely_obstacles.append(
                    geometry.Polygon(
                        [
                            [obs.v1.x, obs.v1.y],
                            [obs.v2.x, obs.v2.y],
                            [obs.v3.x, obs.v3.y],
                        ]
                    )
                )
            elif isinstance(obs, Circle):
                shapely_obstacles.append(geometry.Point(obs.center.x, obs.center.y).buffer(obs.radius))
            else:
                raise ValueError("Obstacle must be Polygon, Triangle, or Circle")

        # Check distance between each line segment with each polygon
        for i, (p1, p2) in enumerate(zip(path.waypoints[:-1], path.waypoints[1:])):
            ls_shapely = geometry.LineString([geometry.Point(p1.x, p1.y), geometry.Point(p2.x, p2.y)])
            for obs in shapely_obstacles:
                if ls_shapely.distance(obs) <= r:
                    ground_truth.append(i)
                    break

        return path, r, obstacles, ground_truth

    @staticmethod
    def generate_robot_frame_data(
        index: int,
    ) -> tuple[
        list[SE2Transform],
        float,
        list[list[GeoPrimitive]],
        list[GeoPrimitive],
        list[int],
    ]:
        # Initialize Random Map
        (
            path,
            r,
            obstacles,
            ground_truth,
        ) = DataGenerator.generate_random_robot_map_and_path(11, index)
        # Calculate Robot Poses
        # In every waypoint robot will turn into its next position
        poses = []
        theta = 0
        for wp_1, wp_2 in zip(path.waypoints[:-1], path.waypoints[1:]):
            wp_temp = Point(wp_2.x - wp_1.x, wp_2.y - wp_1.y)
            theta = np.arctan2(wp_temp.y, wp_temp.x)

            poses.append(SE2Transform((wp_1.x, wp_1.y), theta))
        # Append Last Pose with default last theta
        last_point = path.waypoints[-1]
        poses.append(SE2Transform((last_point.x, last_point.y), theta))
        # Calculate Observed Obstacles
        observation_radius = 50
        # Convert obstacles to Shapely Shapes
        shapely_obstacles = []
        for obs in obstacles:
            if isinstance(obs, Polygon):
                shapely_obstacles.append(geometry.Polygon([[p.x, p.y] for p in obs.vertices]))
            elif isinstance(obs, Triangle):
                shapely_obstacles.append(
                    geometry.Polygon(
                        [
                            [obs.v1.x, obs.v1.y],
                            [obs.v2.x, obs.v2.y],
                            [obs.v3.x, obs.v3.y],
                        ]
                    )
                )
            elif isinstance(obs, Circle):
                shapely_obstacles.append(geometry.Point(obs.center.x, obs.center.y).buffer(obs.radius))
            else:
                raise ValueError("Obstacle must be Polygon, Triangle, or Circle")
        observations = []
        for pose_index, pose in enumerate(poses):
            observations.append([])
            sensing_distance = observation_radius
            if pose_index + 1 < len(poses):
                sensing_distance = max(
                    sensing_distance,
                    float(np.linalg.norm(poses[pose_index + 1].p - pose.p)),
                )
            # Check distance to obstacles
            shapely_point = geometry.Point(pose.p[0], pose.p[1])
            for shapely_obs, obs in zip(shapely_obstacles, obstacles):
                if shapely_point.distance(shapely_obs) <= sensing_distance + r:
                    # Calculate position of the obstacle in robot frame
                    robot_frame_poly = obs.apply_SE2transform(inv(pose.as_SE2()))
                    observations[-1].append(robot_frame_poly)

        return poses, r, observations, obstacles, ground_truth
