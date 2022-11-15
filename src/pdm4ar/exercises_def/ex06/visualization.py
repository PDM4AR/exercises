from pdm4ar.exercises_def.ex06.data import *
from numpy.linalg import inv
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF


def visualize_circle_point(r: Report, ex_num: str, data: Tuple[Circle, Point, bool]):
    c, p, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"point-circle-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Point
        p.visualize(ax)

        # Draw Circle
        c.visualize(ax)

        ax.set_xlim(
            min(c.center.x - c.radius, p.x) - 1,
            max(c.center.x + c.radius, p.x) + 1,
        )
        ax.set_ylim(
            min(c.center.y - c.radius, p.y) - 1,
            max(c.center.y + c.radius, p.y) + 1,
        )


def visualize_triangle_point(
    r: Report, ex_num: str, data: Tuple[Triangle, Point, bool]
):
    t, p, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"point-triangle-primitive-{ex_num}",
        mime=MIME_PDF,
        figsize=None,
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Point
        p.visualize(ax)

        # Draw Triangle
        t.visualize(ax)

        ax.set_xlim(
            min([t.v1.x, t.v2.x, t.v3.x, p.x]) - 1,
            max([t.v1.x, t.v2.x, t.v3.x, p.x]) + 1,
        )
        ax.set_ylim(
            min([t.v1.y, t.v2.y, t.v3.y, p.y]) - 1,
            max([t.v1.y, t.v2.y, t.v3.y, p.y]) + 1,
        )


def visualize_polygon_point(r: Report, ex_num: str, data: Tuple[Polygon, Point, bool]):
    poly, p, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"point-polygon-primitive-{ex_num}",
        mime=MIME_PDF,
        figsize=None,
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Point
        p.visualize(ax)

        # Draw Polygon
        poly.visualize(ax)

        ax.set_xlim(
            min([c.x for c in poly.vertices] + [p.x]) - 1,
            max([c.x for c in poly.vertices] + [p.x]) + 1,
        )
        ax.set_ylim(
            min([c.y for c in poly.vertices] + [p.y]) - 1,
            max([c.y for c in poly.vertices] + [p.y]) + 1,
        )


def visualize_circle_line(r: Report, ex_num: str, data: Tuple[Circle, Segment, bool]):
    c, l, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"segment-circle-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Segment
        l.visualize(ax)

        # Draw Circle
        c.visualize(ax)

        ax.set_xlim(
            min([c.center.x - c.radius, l.p1.x, l.p2.x]) - 1,
            max([c.center.x + c.radius, l.p1.x, l.p2.x]) + 1,
        )
        ax.set_ylim(
            min([c.center.y - c.radius, l.p1.y, l.p2.y]) - 1,
            max([c.center.y + c.radius, l.p1.y, l.p2.y]) + 1,
        )


def visualize_triangle_line(
    r: Report, ex_num: str, data: Tuple[Triangle, Segment, bool]
):
    t, l, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"segment-triangle-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Segment
        l.visualize(ax)

        # Draw Circle
        t.visualize(ax)

        ax.set_xlim(
            min([t.v1.x, t.v2.x, t.v3.x, l.p1.x, l.p2.x]) - 1,
            max([t.v1.x, t.v2.x, t.v3.x, l.p1.x, l.p2.x]) + 1,
        )
        ax.set_ylim(
            min([t.v1.y, t.v2.y, t.v3.y, l.p1.y, l.p2.y]) - 1,
            max([t.v1.y, t.v2.y, t.v3.y, l.p1.y, l.p2.y]) + 1,
        )


def visualize_polygon_line(r: Report, ex_num: str, data: Tuple[Polygon, Segment, bool]):
    poly, l, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"segment-polygon-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Segment
        l.visualize(ax)

        # Draw Polygon
        poly.visualize(ax)

        ax.set_xlim(
            min([c.x for c in poly.vertices] + [l.p1.x, l.p2.x]) - 1,
            max([c.x for c in poly.vertices] + [l.p1.x, l.p2.x]) + 1,
        )
        ax.set_ylim(
            min([c.y for c in poly.vertices] + [l.p1.y, l.p2.y]) - 1,
            max([c.y for c in poly.vertices] + [l.p1.y, l.p2.y]) + 1,
        )


def visualize_map_path(
    r: Report,
    ex_num: str,
    data: Tuple[Path, float, List[GeoPrimitive], List[int]],
):
    path, radius, obstacles, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"map-path-collision-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Segment
        path.visualize(ax)

        for obs in obstacles:
            obs.visualize(ax)
        ax.set_aspect(1)

        boundaries = [path.get_boundaries()] + [
            obs.get_boundaries() for obs in obstacles
        ]

        ax.set_xlim(
            min([p_min.x for p_min, _ in boundaries]) - 1,
            max([p_max.x for _, p_max in boundaries]) + 1,
        )
        ax.set_ylim(
            min([p_min.y for p_min, _ in boundaries]) - 1,
            max([p_max.y for _, p_max in boundaries]) + 1,
        )


def visualize_robot_frame_map(
    r: Report,
    ex_num: str,
    data: Tuple[
        List[SE2Transform],
        float,
        List[List[GeoPrimitive]],
        List[GeoPrimitive],
        List[int],
    ],
):
    path, _, observations, obstacles, _ = data

    # Visualize Map
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"map-path-collision-robot-frame-{ex_num}",
        mime=MIME_PDF,
        figsize=None,
    ) as _:
        ax = plt.gca()
        ax.grid()

        x_values = [p.p[0] for p in path]
        y_values = [p.p[1] for p in path]

        # Draw Segment
        ax.plot(x_values, y_values, "gx--", markersize=15)

        for poly in obstacles:
            # Draw Polygon
            poly.visualize(ax)
        ax.set_aspect(1)

        boundaries = [obs.get_boundaries() for obs in obstacles]

        ax.set_xlim(
            min(x_values + [p_min.x for p_min, _ in boundaries]) - 1,
            max(x_values + [p_max.x for _, p_max in boundaries]) + 1,
        )
        ax.set_ylim(
            min(y_values + [p_min.y for p_min, _ in boundaries]) - 1,
            max(y_values + [p_max.y for _, p_max in boundaries]) + 1,
        )

    # Visualize Inputs Step by Step
    for i, (pose, observation) in enumerate(zip(path[:-1], observations[:-1])):
        # If there is not any observation ignore the step
        if len(observation) == 0:
            continue

        rfig = r.figure(cols=1)
        with rfig.plot(
            nid=f"map-path-collision-robot-frame-{ex_num}-step{i}",
            mime=MIME_PDF,
            figsize=None,
        ) as _:
            ax = plt.gca()
            ax.grid()
            # express next goal in current pose frame
            new_goal = SE2Transform.from_SE2(inv(pose.as_SE2()) @ path[i + 1].as_SE2())

            for obs in observation:
                # Draw Polygon
                obs.visualize(ax)

            # Draw Point
            segment = Segment(Point(0, 0), Point(*new_goal.p))
            segment.visualize(ax)
            ax.set_aspect(1)

            boundaries = [segment.get_boundaries()] + [
                obs.get_boundaries() for obs in observation
            ]

            ax.set_xlim(
                min([p_min.x for p_min, _ in boundaries]) - 1,
                max([p_max.x for _, p_max in boundaries]) + 1,
            )
            ax.set_ylim(
                min([p_min.y for p_min, _ in boundaries]) - 1,
                max([p_max.y for _, p_max in boundaries]) + 1,
            )
