from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

DEFAULT_BOUNDARY: list[list[float]] = [
    [-11.0, -11.0],
    [-11.0, 11.0],
    [11.0, 11.0],
    [11.0, -11.0],
    [-11.0, -11.0],
]
DEFAULT_AGENT_COLORS = [
    "saddlebrown",
    "steelblue",
    "firebrick",
    "seagreen",
    "indigo",
    "darkorange",
    "goldenrod",
    "darkslategray",
]


@dataclass(frozen=True)
class GenerationParams:
    """Tunables for random config generation."""

    robot_width: float = 1.2
    margin: float = 0.3
    goal_radius: float = 0.3
    collection_radius: float = 0.8
    obstacle_size_range: tuple[float, float] = (2.0, 10.0)
    max_attempts: int = 1000


def generate_random_config(
    num_agents: int,
    num_goals: int,
    num_collection_points: int,
    num_obstacles: int,
    boundary: Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
    params: GenerationParams | None = None,
    config_name: str | None = None,
) -> Mapping[str, Any]:
    """
    Randomly generate a valid ex14 config with geometric clearance checks.
    Ensures agents, goals and collection points are not inside or fully blocked by obstacles.
    Obstacles keep a clearance to the boundary and to one another of at least (robot_width + margin).
    """

    if params is None:
        params = GenerationParams()
    rng = random.Random(seed)

    # Derived clearances
    corridor_clearance = params.robot_width + params.margin * 2
    footprint_clearance = params.robot_width * 0.5 + params.margin

    boundary_coords = _ensure_closed_boundary(boundary or DEFAULT_BOUNDARY)
    boundary_poly = _as_polygon(boundary_coords)

    obstacles = _sample_obstacles(
        num_obstacles=num_obstacles,
        boundary=boundary_poly,
        corridor_clearance=corridor_clearance,
        size_range=params.obstacle_size_range,
        rng=rng,
        max_attempts=params.max_attempts,
    )

    placed_points: list[tuple[Point, float]] = []

    collection_points = []
    for idx in range(num_collection_points):
        cp_radius = params.collection_radius + footprint_clearance
        location = _sample_point(
            boundary=boundary_poly,
            obstacles=obstacles,
            candidate_radius=cp_radius,
            rng=rng,
            max_attempts=params.max_attempts,
            other_points=placed_points,
        )
        placed_points.append((location, cp_radius))
        collection_points.append(
            {
                "id": f"collection_{idx + 1}",
                "center": [location.x, location.y],
                "radius": params.collection_radius,
            }
        )

    shared_goals = []
    for idx in range(num_goals):
        goal_radius = params.goal_radius + footprint_clearance
        location = _sample_point(
            boundary=boundary_poly,
            obstacles=obstacles,
            candidate_radius=goal_radius,
            rng=rng,
            max_attempts=params.max_attempts,
            other_points=placed_points,
        )
        placed_points.append((location, goal_radius))
        shared_goals.append(
            {
                "id": f"goal_{idx + 1}",
                "center": [location.x, location.y],
                "radius": params.goal_radius,
            }
        )

    agents = {}
    agent_positions: list[tuple[Point, float]] = []
    for idx in range(num_agents):
        agent_radius = footprint_clearance
        location = _sample_point(
            boundary=boundary_poly,
            obstacles=obstacles,
            candidate_radius=agent_radius,
            rng=rng,
            max_attempts=params.max_attempts,
            other_points=placed_points + agent_positions,
        )
        agent_positions.append((location, agent_radius))


        agents[f"PDM4AR_{idx + 1}"] = {
            "state": {"x": location.x, "y": location.y, "psi": rng.uniform(-math.pi, math.pi)},
            "color": DEFAULT_AGENT_COLORS[idx % len(DEFAULT_AGENT_COLORS)],
        }

    config: dict[str, Any] = {
        "agents": agents,
        "static_obstacles": [_polygon_to_coords(o) for o in obstacles],
        "boundary": boundary_coords,
        "seed": seed if seed is not None else rng.randint(0, 10**6),
    }
    if shared_goals:
        config["shared_goals"] = shared_goals
    if collection_points:
        config["collection_points"] = collection_points
    if config_name:
        config["config_name"] = config_name

    validate_config(config=config, robot_width=params.robot_width, margin=params.margin)
    return config


def validate_config(config: Mapping[str, Any], robot_width: float, margin: float) -> None:
    """Raise a ValueError if the config violates clearance rules."""
    corridor_clearance = robot_width + margin
    footprint_clearance = robot_width * 0.5 + margin
    boundary_poly = _as_polygon(_ensure_closed_boundary(config["boundary"]))
    obstacle_polys = [_as_polygon(o) for o in config.get("static_obstacles", [])]
    _assert_obstacle_clearances(obstacle_polys, boundary_poly, corridor_clearance)

    agents = config.get("agents", {})
    for pn, agent in agents.items():
        pos = Point(agent["state"]["x"], agent["state"]["y"])
        _assert_point_clear(pos, footprint_clearance, boundary_poly, obstacle_polys, label=f"agent {pn}")

    for entry in config.get("shared_goals", []):
        pos = Point(entry["center"])
        radius = entry.get("radius", 0.0) + footprint_clearance
        _assert_point_clear(pos, radius, boundary_poly, obstacle_polys, label=f"shared_goal {entry.get('id')}")

    for entry in config.get("collection_points", []):
        pos = Point(entry["center"])
        radius = entry.get("radius", 0.0) + footprint_clearance
        _assert_point_clear(pos, radius, boundary_poly, obstacle_polys, label=f"collection_point {entry.get('id')}")

    _assert_mutual_point_distance(
        points=[(Point(a["state"]["x"], a["state"]["y"]), footprint_clearance) for a in agents.values()],
        label="agents",
    )


def save_config_to_yaml(config: Mapping[str, Any], file_path: str | Path) -> None:
    """Persist a generated config to disk."""
    path = Path(file_path)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


# --- Helpers ---


def _sample_obstacles(
    num_obstacles: int,
    boundary: Polygon,
    corridor_clearance: float,
    size_range: tuple[float, float],
    rng: random.Random,
    max_attempts: int,
) -> list[Polygon]:
    obstacles: list[Polygon] = []
    if num_obstacles == 0:
        return obstacles

    min_size, max_size = size_range
    if min_size <= 0 or max_size <= 0 or max_size < min_size:
        raise ValueError("Invalid obstacle size range")

    # max_radius = max_size * 0.5
    # inner_boundary = boundary.buffer(-(corridor_clearance + max_radius))
    if boundary.buffer(-(corridor_clearance + max_size * 0.5)).is_empty:
        raise ValueError("Boundary too small to place obstacles with the requested clearance.")

    for _ in range(num_obstacles):
        for _ in range(max_attempts):
            radius = rng.uniform(min_size, max_size) * 0.5
            inner_boundary = boundary.buffer(-(radius))
            if inner_boundary.is_empty:
                continue
            cx = rng.uniform(inner_boundary.bounds[0], inner_boundary.bounds[2])
            cy = rng.uniform(inner_boundary.bounds[1], inner_boundary.bounds[3])

            # random convex-ish polygon (3-6 sides) with a random rotation
            candidate_shape = _random_convex_polygon(rng, radius=radius)
            if candidate_shape is None:
                continue
            candidate_shape = translate(candidate_shape, xoff=cx, yoff=cy)
            candidate_shape = rotate(candidate_shape, rng.uniform(0, 360), origin=(cx, cy))

            if not candidate_shape.is_valid or candidate_shape.is_empty:
                continue
            if not inner_boundary.contains(candidate_shape):
                continue
            if boundary.exterior.distance(candidate_shape) < corridor_clearance:
                continue
            if any(candidate_shape.distance(o) < corridor_clearance for o in obstacles):
                continue
            obstacles.append(candidate_shape)
            break
        else:
            raise RuntimeError("Failed to place all obstacles without violating clearance constraints.")

    return obstacles


def _sample_point(
    boundary: Polygon,
    obstacles: Sequence[BaseGeometry],
    candidate_radius: float,
    rng: random.Random,
    max_attempts: int,
    other_points: Iterable[tuple[Point, float]] = (),
) -> Point:
    search_area = boundary.buffer(-candidate_radius)
    if search_area.is_empty:
        raise ValueError("Boundary too tight for the requested clearance.")

    bounds = search_area.bounds
    for _ in range(max_attempts):
        px = rng.uniform(bounds[0], bounds[2])
        py = rng.uniform(bounds[1], bounds[3])
        candidate = Point(px, py)
        if not search_area.contains(candidate):
            continue
        if any(candidate.distance(o) < candidate_radius for o in obstacles):
            continue
        if any(candidate.distance(p) < candidate_radius + r for p, r in other_points):
            continue
        return candidate

    raise RuntimeError("Could not sample a free point with the requested clearance.")


def _assert_obstacle_clearances(obstacles: Sequence[Polygon], boundary: Polygon, min_clearance: float) -> None:
    for obs in obstacles:
        if not obs.is_valid:
            raise ValueError("Invalid obstacle geometry detected.")
        if not boundary.contains(obs):
            raise ValueError("Obstacle lies outside of the boundary.")
        if boundary.exterior.distance(obs) < min_clearance:
            raise ValueError("Obstacle too close to boundary.")
    for i, obs in enumerate(obstacles):
        for other in obstacles[i + 1 :]:
            if obs.distance(other) < min_clearance:
                raise ValueError("Obstacles are too close to one another.")


def _assert_point_clear(
    point: Point,
    radius: float,
    boundary: Polygon,
    obstacles: Sequence[BaseGeometry],
    label: str,
) -> None:
    if not boundary.buffer(-radius).contains(point):
        raise ValueError(f"{label} is too close to the boundary.")
    for obs in obstacles:
        if point.distance(obs) < radius:
            raise ValueError(f"{label} is colliding with or blocked by an obstacle.")


def _assert_mutual_point_distance(points: Sequence[tuple[Point, float]], label: str) -> None:
    for idx, (p, clearance) in enumerate(points):
        for op, o_clearance in points[idx + 1 :]:
            if p.distance(op) < clearance + o_clearance:
                raise ValueError(f"{label} overlap at placement.")


def _polygon_to_coords(poly: Polygon) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly.exterior.coords]


def _ensure_closed_boundary(boundary: Sequence[Sequence[float]]) -> list[list[float]]:
    coords = [[float(x), float(y)] for x, y in boundary]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _as_polygon(coords: Sequence[Sequence[float]]) -> Polygon:
    poly = Polygon(coords)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("Boundary/obstacle polygon is invalid.")
    return poly


def _random_convex_polygon(
    rng: random.Random, radius: float, min_vertices: int = 3, max_vertices: int = 6
) -> Polygon | None:
    if min_vertices < 3 or max_vertices < min_vertices:
        raise ValueError("Invalid vertex count for obstacle polygons.")
    n = rng.randint(min_vertices, max_vertices)
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))
    pts = []
    for a in angles:
        r = rng.uniform(0.5 * radius, radius)
        pts.append((r * math.cos(a), r * math.sin(a)))
    poly = Polygon(pts)
    # Convexify the polygon by taking its convex hull
    poly = poly.convex_hull
    if poly.is_empty or not poly.is_valid or poly.area <= 0:
        return None
    return poly


if __name__ == "__main__":
    # Example usage when running the module directly
    random_config = generate_random_config(
        num_agents=3,
        num_goals=12,
        num_collection_points=2,
        num_obstacles=10,
        seed=42,
        config_name="random_ex14_config",
    )
    target_path = Path(__file__).resolve().parent / "random_ex14_config.yaml"
    save_config_to_yaml(random_config, target_path)
    print(f"Generated {target_path}")
