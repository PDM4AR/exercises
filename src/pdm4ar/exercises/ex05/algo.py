from collections.abc import Sequence

from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    return DubinsParam(min_radius=0)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    dummy_circle = Curve.create_circle(
        center=SE2Transform.identity(),
        config_on_circle=SE2Transform.identity(),
        radius=0.1,
        curve_type=DubinsSegmentType.LEFT,
    )  # TODO remove
    return TurningCircle(left=dummy_circle, right=dummy_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
    # TODO implement here your solution
    return []  # i.e., [Line(),...]


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    return []  # e.g., [Curve(), Line(),..]


def compare_spline_to_dubins(
    start_config: SE2Transform, end_config: SE2Transform, radius: float
) -> tuple[float, float, bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the Dubins path and a cubic Hermite spline between two configurations.

    Returns:
        dubins_length: optimal Dubins path length
        spline_length: numerical length of the Hermite spline
        is_feasible: True if spline curvature ≤ 1 / radius everywhere
    """
    # TODO implement here your solution
    dubins_length = 0.0  # Replace with actual Dubins path length calculation
    # Generate a cubic Hermite spline between start and end configurations (find the parameters)
    # Important: Scale t0,t1 by the direction
    p0 = np.zeros(2, dtype=float)
    p1 = np.zeros(2, dtype=float)
    direction = p1 - p0
    scale = np.linalg.norm(direction)
    t0 = np.zeros(2, dtype=float)
    t1 = np.zeros(2, dtype=float)

    # Hermite basis (parameter s ∈ [0, 1])
    def hermite(s):
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2
        return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

    spline_length = 0.0  # Replace with actual spline length calculation
    is_feasible = True  # Replace with actual feasibility check
    return dubins_length, spline_length, is_feasible, t0, t1, p0, p1


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    return []  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]
