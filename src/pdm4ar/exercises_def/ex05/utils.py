from tkinter import SE
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dg_commons import SE2Transform
from matplotlib.lines import Line2D, segment_hits
from matplotlib.patches import Circle
from pdm4ar.exercises.ex05.structures import DubinsSegmentType, Curve, Line, Path


def get_rot_matrix(alpha: float) -> np.ndarray:
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_matrix

def get_next_point_on_curve(curve: Curve,  point: SE2Transform, delta_angle: float) -> SE2Transform:
    point_translated = point.p - curve.center.p
    rot_matrix = get_rot_matrix(delta_angle)
    next_point = SE2Transform((rot_matrix @ point_translated) + curve.center.p, point.theta + delta_angle)
    return next_point

def get_next_point_on_line(line: Line, point: SE2Transform, delta_length: float) -> SE2Transform:
    return SE2Transform(point.p + delta_length*line.direction)
 
def interpolate_line_points(line: Line, number_of_points: float) -> List[SE2Transform]:
    start = line.start_config
    end = line.end_config
    start_to_end = end.p - start.p
    intervals = np.linspace(0, 1.0, number_of_points)
    return [SE2Transform(start.p + i*start_to_end , start.theta) for i in intervals]

def interpolate_curve_points(curve: Curve, number_of_points: float)-> List[SE2Transform]:
    pts_list = []
    angle = curve.arc_angle
    direction = curve.type
    angle = curve.gear.value * direction.value * angle
    split_angle = angle / number_of_points
    old_point = curve.start_config
    for i in range(number_of_points):
        pts_list.append(old_point)
        point_next = get_next_point_on_curve(curve, point = old_point, delta_angle = split_angle)
        old_point = point_next
    return pts_list

def extract_path_points(path: Path) -> List[SE2Transform]:
    """ Extracts a fixed number of SE2Transform points on a path"""
    pts_list = []
    num_points_per_segment = 20
    for idx, seg in enumerate(path):
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
        ax.axis('equal')
    # Plot heading
    arrow_freq = 5
    ax.quiver(pts_array[::arrow_freq, 0], pts_array[::arrow_freq, 1], np.cos(pts_array[::arrow_freq, 2]),
              np.sin(pts_array[::arrow_freq, 2]))
    # Plot trajectory
    ax.plot(pts_array[:, 0], pts_array[:, 1])
    return ax


def plot_circle(circle: Curve, ax: matplotlib.axes.Axes):
    _circle = Circle((circle.center.p[0], circle.center.p[1]), circle.radius,
                     ec="tab:blue" if circle.type is DubinsSegmentType.LEFT else "tab:red", fill=False)
    ax.add_patch(_circle)


def plot_configuration(config: SE2Transform, ax: matplotlib.axes.Axes, color='tab:green'):
    ax.quiver(config.p[0], config.p[1], np.cos(config.theta), np.sin(config.theta), color=color)


def plot_circle_tangents(circle1, circle2, tan_list, draw_heading=True,
                         ax: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect(1)
    plot_circle(circle1, ax)
    plot_circle(circle2, ax)

    for tangent in tan_list:
        t_start = tangent.start_config.p
        t_end = tangent.end_config.p
        tangent_line = Line2D([t_start[0], t_end[0]], [t_start[1], t_end[1]], color='black', ls='--')
        #center_1_line = Line2D([circle1.center.p[0], t_start[0]], [circle1.center.p[1], t_start[1]])
        #center_2_line = Line2D([circle2.center.p[0], t_end[0]], [circle2.center.p[1], t_end[1]])
        ax.add_line(tangent_line)
        if draw_heading:
            ax.quiver(t_start[0], t_start[1], np.cos(tangent.start_config.theta), np.sin(tangent.start_config.theta))
            ax.quiver(t_end[0], t_end[1], np.cos(tangent.end_config.theta), np.sin(tangent.end_config.theta))
    return ax


if __name__ == "__main__":
    # Plot
    matplotlib.use('tkagg')  # change to MacOSX if want to run on mac os
    test_config = SE2Transform([1.0, 2], 3)
    fig, ax = plt.subplots()
    plot_configuration(test_config, ax)
    plt.show()
