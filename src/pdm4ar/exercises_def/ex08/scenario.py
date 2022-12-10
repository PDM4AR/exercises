import random
from typing import Optional, Tuple

from dg_commons import apply_SE2_to_shapely_geo, X
from dg_commons.maps.shapes_generator import create_random_starshaped_polygon
from dg_commons.planning import PlanningGoal, PolygonGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios.structures import DgScenario
from geometry import SE2_from_xytheta
from numpy import deg2rad
from shapely.geometry import Polygon, LinearRing

__all__ = ["get_dgscenario"]


def get_dgscenario(seed: Optional[int] = None) -> Tuple[DgScenario, PlanningGoal, X]:
    max_size = 100
    shapes = []
    bounds = LinearRing([(0, 0), (0, max_size), (max_size, max_size), (max_size, 0), (0, 0)])
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))
    shapes += [bounds, poly2]
    if seed is not None:
        random.seed(seed)

    positions = [(50, 50), (50, 10), (10, 50), (10, 85), (75, 80), (80, 30), ]
    for pos in positions:
        poly = Polygon(create_random_starshaped_polygon(*pos, 10, 0.5, 0.5, 10))
        shapes.append(poly)

    obstacles = list(map(StaticObstacle, shapes))
    static_obstacles = dict(zip(range(len(obstacles)), obstacles))

    x0 = VehicleState(x=7, y=4, psi=deg2rad(60), vx=5, delta=-0.02)
    goal_poly = Polygon(((max_size, max_size), (max_size - 10, max_size),
                         (max_size - 10, max_size - 10), (max_size, max_size - 10)))
    goal = PolygonGoal(goal_poly)
    dg_scenario = DgScenario(static_obstacles=static_obstacles)

    return dg_scenario, goal, x0
