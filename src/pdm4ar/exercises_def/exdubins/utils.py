from typing import List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dg_commons import SE2Transform
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pdm4ar.exercises.exdubins.structures import DubinsSegmentType, Curve, Line, Path


def get_rot_matrix(alpha: float) -> np.ndarray:
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return rot_matrix


def get_next_point_on_curve(center: SE2Transform, dangle: float, current_point: SE2Transform) -> SE2Transform:
    point_translated = current_point.p - center.p
    rot_matrix = get_rot_matrix(dangle)
    next_point = SE2Transform((rot_matrix @ point_translated) + center.p, current_point.theta + dangle)
    return next_point


def extract_path_points(path: Path) -> List[SE2Transform]:
    pts_list = []
    num_curve_pts = 20

    for idx, seg in enumerate(path):
        if seg.type is DubinsSegmentType.STRAIGHT:
            pts_list.append(seg.start_config)
        else:  # curve
            curve = seg
            center = curve.center
            angle = curve.arc_angle
            direction = curve.type
            angle = curve.gear.value * direction.value * angle
            split_angle = angle / num_curve_pts
            old_point = seg.start_config
            for i in range(num_curve_pts):
                pts_list.append(old_point)
                point_next = get_next_point_on_curve(center, split_angle, old_point)
                old_point = point_next
                # print(old_point)
                if i == num_curve_pts - 1:
                    break

        if idx == len(path) - 1:
            pts_list.append(seg.end_config)
            # print(f'end {seg.end_config}')

    return pts_list


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
    # ax.set_xlim((-10, 10))
    # ax.set_ylim((-10, 10))
    plot_circle(circle1, ax)
    plot_circle(circle2, ax)
    # ax.scatter(pt_1[0], pt_1[1], color='tab:red')
    # ax.scatter(pt_2[0], pt_2[1], color='tab:red')

    for tangent in tan_list:
        t_start = tangent.start_config.p
        t_end = tangent.end_config.p
        tangent_line = Line2D([t_start[0], t_end[0]], [t_start[1], t_end[1]], color='black', ls='--')
        center_1_line = Line2D([circle1.center.p[0], t_start[0]], [circle1.center.p[1], t_start[1]])
        center_2_line = Line2D([circle2.center.p[0], t_end[0]], [circle2.center.p[1], t_end[1]])

        ax.add_line(tangent_line)
        # ax.add_line(center_1_line)
        # ax.add_line(center_2_line)

        if draw_heading:
            ax.quiver(t_start[0], t_start[1], np.cos(tangent.start_config.theta), np.sin(tangent.start_config.theta))
            ax.quiver(t_end[0], t_end[1], np.cos(tangent.end_config.theta), np.sin(tangent.end_config.theta))

    # plt.show()
    return ax


if __name__ == "__main__":
    # Plot
    matplotlib.use('tkagg')  # change to MacOSX if want to run on mac os
    test_config = SE2Transform([1.0, 2], 3)
    fig, ax = plt.subplots()
    plot_configuration(test_config, ax)
    plt.show()
