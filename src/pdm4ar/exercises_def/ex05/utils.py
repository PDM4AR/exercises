from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dg_commons import SE2Transform
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pdm4ar.exercises.ex05.structures import DubinsSegmentType, Curve, Line, Path, mod_2_pi


def get_rot_matrix(alpha: float) -> np.ndarray:
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_matrix


def get_next_point_on_curve(curve: Curve, point: SE2Transform, delta_angle: float) -> SE2Transform:
    point_translated = point.p - curve.center.p
    rot_matrix = get_rot_matrix(delta_angle)
    next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)
    return next_point


def get_next_point_on_line(line: Line, point: SE2Transform, delta_length: float) -> SE2Transform:
    return SE2Transform(point.p + delta_length * line.direction, theta=point.theta)


def interpolate_line_points(line: Line, number_of_points: float) -> List[SE2Transform]:
    start = line.start_config
    end = line.end_config
    start_to_end = end.p - start.p
    intervals = np.linspace(0, 1.0, number_of_points)
    return [SE2Transform(start.p + i * start_to_end, start.theta) for i in intervals]


def interpolate_curve_points(curve: Curve, number_of_points: float) -> List[SE2Transform]:
    pts_list = []
    angle = curve.arc_angle
    direction = curve.type
    angle = curve.gear.value * direction.value * angle
    split_angle = angle / number_of_points
    old_point = curve.start_config
    for i in range(number_of_points):
        pts_list.append(old_point)
        point_next = get_next_point_on_curve(curve, point=old_point, delta_angle=split_angle)
        old_point = point_next
    return pts_list


def update_arc_length(curve: Curve) -> None:
    """Updates a curve's arc angle and length in place."""
    reverse_sign = curve.gear.value

    if np.allclose(np.linalg.norm(curve.start_config.p - curve.end_config.p), 0.0):
        curve.arc_angle = 0.0
        curve.length = curve.radius * curve.arc_angle
        return

    center_to_start = curve.start_config.p - curve.center.p
    center_to_end = curve.end_config.p - curve.center.p
    cos_val = np.clip(center_to_start.T @ center_to_end / (curve.radius**2), -1.0, 1.0)
    alpha = np.arccos(cos_val)
    if np.sign(np.cross(center_to_start, center_to_end)) * reverse_sign * curve.type.value == -1:  # TODO check again
        alpha = 2 * np.pi - alpha
    curve.arc_angle = mod_2_pi(alpha)
    curve.length = curve.radius * curve.arc_angle


def get_heading_angle_point_on_curve(curve: Curve, point: np.ndarray) -> float:
    """Computes the heading angle of a point on a curve based on the curve's center and type"""
    n = point - curve.center.p
    heading_vector = curve.gear.value * curve.type.value * np.array([-n[1], n[0]])
    theta = float(np.arctan2(heading_vector[1], heading_vector[0]))
    return theta


def compute_middle_curve(circle_start: Curve, circle_end: Curve, radius: float) -> list[Curve]:
    """Computes the middle turning arc connecting two turning circles of equal radius based on their centers and types.
    This helper is used in the construction of CCC-type Dubins/Reeds-Shepp paths"""
    if radius <= 0:
        raise ValueError("The radius must be positive.")
    if circle_start.gear != circle_end.gear:
        raise ValueError("The start and end circles must use the same gear.")
    if circle_start.type != circle_end.type:
        raise ValueError("CCC paths require start and end circles of the same type.")
    if not np.isclose(circle_start.radius, radius) or not np.isclose(circle_end.radius, radius):
        raise ValueError("The circle radii must match the provided radius.")

    # Two approaches:
    # Simple : consider triangle and cosine law (done here)
    # Alternative compute intersection of 2r circles to get center points, then compute intersection point
    start_to_end = circle_end.center.p - circle_start.center.p
    distance_circles = np.linalg.norm(start_to_end)
    if np.allclose(distance_circles, 0):
        return []
    start_to_end_norm = start_to_end / distance_circles
    if distance_circles > 4 * radius and not np.isclose(distance_circles, 4 * radius, rtol=1e-9, atol=1e-12):
        return []

    middle_curve_type = DubinsSegmentType(-1 * circle_start.type.value)  # Take opposite direction
    theta = np.arccos(np.clip(distance_circles / (4 * radius), -1.0, 1.0))

    signs = [1] if np.isclose(theta, 0.0) else [1, -1]
    middle_curve_list = []

    for sign in signs:
        rot_matrix = get_rot_matrix(sign * theta)
        n = rot_matrix @ start_to_end_norm

        middle_center = SE2Transform(circle_start.center.p + 2 * radius * n, 0)
        intersection_pt_1_xy = circle_start.center.p + radius * n
        intersection_pt_2_xy = middle_center.p + radius * (circle_end.center.p - middle_center.p) / (
            np.linalg.norm(circle_end.center.p - middle_center.p)
        )
        intersection_pt_1_theta = get_heading_angle_point_on_curve(circle_start, intersection_pt_1_xy)
        intersection_pt_2_theta = get_heading_angle_point_on_curve(circle_end, intersection_pt_2_xy)
        middle_config_1 = SE2Transform(intersection_pt_1_xy, intersection_pt_1_theta)
        middle_config_2 = SE2Transform(intersection_pt_2_xy, intersection_pt_2_theta)

        # middle_curve has the same gear attribute

        # Construct the middle curve
        middle_curve = Curve(
            start_config=middle_config_1,
            end_config=middle_config_2,
            center=middle_center,
            radius=radius,
            arc_angle=0,  # Set appropriate arc angle if needed
            curve_type=middle_curve_type,
            gear=circle_start.gear,  # Inherit gear from start (or end) curve
        )

        # middle_curve = Curve(start_config=middle_config_1, end_config=middle_config_2, center=middle_center,
        #  radius=radius, arc_angle=0, curve_type=middle_curve_type)
        update_arc_length(middle_curve)
        middle_curve_list.append(middle_curve)

    return middle_curve_list


def extract_path_points(path: Path) -> List[SE2Transform]:
    """Extracts a fixed number of SE2Transform points on a path"""
    pts_list = []
    num_points_per_segment = 20
    for idx, seg in enumerate(path):
        # if np.allclose(seg.length, 0):
        #     continue
        seg.start_config.theta = mod_2_pi(seg.start_config.theta)
        seg.end_config.theta = mod_2_pi(seg.end_config.theta)
        if seg.type is DubinsSegmentType.STRAIGHT:
            line_pts = interpolate_line_points(seg, num_points_per_segment)
            pts_list.extend(line_pts)
        else:  # Curve
            curve_pts = interpolate_curve_points(seg, num_points_per_segment)
            pts_list.extend(curve_pts)
    pts_list.append(path[-1].end_config)
    return pts_list


def se2_points_to_np_array(se2_list: List[SE2Transform]):
    return np.array([[point.p[0], point.p[1], point.theta] for point in se2_list])


def plot_2d_path(pts_array: np.ndarray, ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.axis("equal")
    # Plot heading
    arrow_freq = 5
    ax.quiver(
        pts_array[::arrow_freq, 0],
        pts_array[::arrow_freq, 1],
        np.cos(pts_array[::arrow_freq, 2]),
        np.sin(pts_array[::arrow_freq, 2]),
    )
    # Plot trajectory
    ax.plot(pts_array[:, 0], pts_array[:, 1])
    return ax


def plot_circle(circle: Curve, ax: matplotlib.axes.Axes):
    _circle = Circle(
        (circle.center.p[0], circle.center.p[1]),
        circle.radius,
        ec="tab:blue" if circle.type is DubinsSegmentType.LEFT else "tab:red",
        fill=False,
    )
    ax.add_patch(_circle)


def plot_configuration(config: SE2Transform, ax: matplotlib.axes.Axes, color="tab:green"):
    ax.quiver(config.p[0], config.p[1], np.cos(config.theta), np.sin(config.theta), color=color)


def plot_circle_tangents(
    circle1, circle2, tan_list, draw_heading=True, ax: matplotlib.axes.Axes = None
) -> matplotlib.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect(1)
    plot_circle(circle1, ax)
    plot_circle(circle2, ax)

    for tangent in tan_list:
        t_start = tangent.start_config.p
        t_end = tangent.end_config.p
        tangent_line = Line2D([t_start[0], t_end[0]], [t_start[1], t_end[1]], color="black", ls="--")
        # center_1_line = Line2D([circle1.center.p[0], t_start[0]], [circle1.center.p[1], t_start[1]])
        # center_2_line = Line2D([circle2.center.p[0], t_end[0]], [circle2.center.p[1], t_end[1]])
        ax.add_line(tangent_line)
        if draw_heading:
            ax.quiver(t_start[0], t_start[1], np.cos(tangent.start_config.theta), np.sin(tangent.start_config.theta))
            ax.quiver(t_end[0], t_end[1], np.cos(tangent.end_config.theta), np.sin(tangent.end_config.theta))
    return ax


if __name__ == "__main__":
    # Plot
    matplotlib.use("tkagg")  # change to MacOSX if want to run on mac os
    test_config = SE2Transform([1.0, 2], 3)
    fig, ax = plt.subplots()
    plot_configuration(test_config, ax)
    plt.show()
