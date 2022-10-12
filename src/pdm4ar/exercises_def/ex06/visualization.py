from pdm4ar.exercises_def.ex06.data import *
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF


def visualize_circle_point(
    r: Report, ex_num: str, data: Tuple[Circle, Point, bool]
):
    c, p, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"point-circle-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Point
        ax.plot(p.x, p.y, marker="x", markersize=10)

        # Draw Circle
        draw_circle = plt.Circle(
            (c.center.x, c.center.y),
            c.radius,
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_circle)

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
        ax.plot(p.x, p.y, marker="x", markersize=10)

        # Draw Triangle
        draw_triangle = plt.Polygon(
            [[t.v1.x, t.v1.y], [t.v2.x, t.v2.y], [t.v3.x, t.v3.y]],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_triangle)

        ax.set_xlim(
            min([t.v1.x, t.v2.x, t.v3.x, p.x]) - 1,
            max([t.v1.x, t.v2.x, t.v3.x, p.x]) + 1,
        )
        ax.set_ylim(
            min([t.v1.y, t.v2.y, t.v3.y, p.y]) - 1,
            max([t.v1.y, t.v2.y, t.v3.y, p.y]) + 1,
        )


def visualize_polygon_point(
    r: Report, ex_num: str, data: Tuple[Polygon, Point, bool]
):
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
        ax.plot(p.x, p.y, marker="x", markersize=10)

        # Draw Polygon
        draw_poly = plt.Polygon(
            [[p.x, p.y] for p in poly.vertices],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_poly)

        ax.set_xlim(
            min([c.x for c in poly.vertices] + [p.x]) - 1,
            max([c.x for c in poly.vertices] + [p.x]) + 1,
        )
        ax.set_ylim(
            min([c.y for c in poly.vertices] + [p.y]) - 1,
            max([c.y for c in poly.vertices] + [p.y]) + 1,
        )


def visualize_circle_line(
    r: Report, ex_num: str, data: Tuple[Circle, Line, bool]
):
    c, l, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"line-circle-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Line
        ax.plot(
            [l.p1.x, l.p2.x], [l.p1.y, l.p2.y], marker="x", markersize=10
        )

        # Draw Circle
        draw_circle = plt.Circle(
            (c.center.x, c.center.y),
            c.radius,
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_circle)

        ax.set_xlim(
            min([c.center.x - c.radius, l.p1.x, l.p2.x]) - 1,
            max([c.center.x + c.radius, l.p1.x, l.p2.x]) + 1,
        )
        ax.set_ylim(
            min([c.center.y - c.radius, l.p1.y, l.p2.y]) - 1,
            max([c.center.y + c.radius, l.p1.y, l.p2.y]) + 1,
        )


def visualize_polygon_line(
    r: Report, ex_num: str, data: Tuple[Polygon, Line, bool]
):
    poly, l, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"line-polygon-primitive-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        # Draw Line
        ax.plot(
            [l.p1.x, l.p2.x], [l.p1.y, l.p2.y], marker="x", markersize=10
        )

        # Draw Polygon
        draw_poly = plt.Polygon(
            [[p.x, p.y] for p in poly.vertices],
            color="r",
            fill=False,
            linewidth=5,
        )
        ax.set_aspect(1)
        ax.add_artist(draw_poly)

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
    data: Tuple[Path, float, List[Polygon], List[int]],
):
    path, radius, obstacles, _ = data
    rfig = r.figure(cols=1)
    with rfig.plot(
        nid=f"map-path-collision-{ex_num}", mime=MIME_PDF, figsize=None
    ) as _:
        ax = plt.gca()
        ax.grid()

        x_values = [p.x for p in path.waypoints]
        y_values = [p.y for p in path.waypoints]

        # Draw Line
        ax.plot(x_values, y_values, "gx--", markersize=15)

        for poly in obstacles:
            # Draw Polygon
            draw_poly = plt.Polygon(
                [[p.x, p.y] for p in poly.vertices], color="r", linewidth=5
            )
            ax.add_artist(draw_poly)
            x_values += [p.x for p in poly.vertices]
            y_values += [p.y for p in poly.vertices]
        ax.set_aspect(1)

        ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
        ax.set_ylim(min(y_values) - 1, max(y_values) + 1)


def visualize_robot_frame_map(
    r: Report,
    ex_num: str,
    data: Tuple[
        List[Pose2D], float, List[List[Polygon]], List[Polygon], List[int]
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

        x_values = [p.position.x for p in path]
        y_values = [p.position.y for p in path]

        # Draw Line
        ax.plot(x_values, y_values, "gx--", markersize=15)

        for poly in obstacles:
            # Draw Polygon
            draw_poly = plt.Polygon(
                [[p.x, p.y] for p in poly.vertices], color="r", linewidth=5
            )
            ax.add_artist(draw_poly)
            x_values += [p.x for p in poly.vertices]
            y_values += [p.y for p in poly.vertices]
        ax.set_aspect(1)

        ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
        ax.set_ylim(min(y_values) - 1, max(y_values) + 1)

    # Visualize Inputs Step by Step
    for i, (pose, observation) in enumerate(zip(path, observations)):
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

            new_goal = (
                path[(i + 1) % len(path)]
                .position.apply_SE2transform(
                    SE2_from_translation_angle(
                        -np.array([pose.position.x, pose.position.y]),
                        0,
                    )
                )
                .apply_SE2transform(
                    SE2_from_translation_angle(
                        np.array([0, 0]),
                        -pose.theta,
                    )
                )
            )

            x_values, y_values = [0, new_goal.x], [0, new_goal.y]

            for poly in observation:
                # Draw Polygon
                draw_poly = plt.Polygon(
                    [[p.x, p.y] for p in poly.vertices],
                    color="r",
                    linewidth=5,
                )
                ax.add_artist(draw_poly)
                x_values += [p.x for p in poly.vertices]
                y_values += [p.y for p in poly.vertices]

            # Draw Point
            ax.plot(
                [0, new_goal.x], [0, new_goal.y], marker="x", markersize=10
            )
            ax.set_aspect(1)

            ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
            ax.set_ylim(min(y_values) - 1, max(y_values) + 1)
