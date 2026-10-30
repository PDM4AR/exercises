"""Student-editable diagnostics using only scenario data and simulator logs.

No planner, reference trajectory, feedback law or agent-specific attributes are
required. Extend these functions to visualize your own solution's diagnostics.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from dg_commons.sim.simulator_visualisation import SimRenderer
from pdm4ar.exercises_def.ex13.goal import DockingTarget


def goal_errors(state, goal):
    """Position, wrapped heading and body-frame velocity errors, as in the goal check."""
    target = goal.target
    angle = state.psi - target.psi
    return np.array([
        np.hypot(state.x - target.x, state.y - target.y),
        abs(np.arctan2(np.sin(angle), np.cos(angle))),
        np.hypot(state.vx - target.vx, state.vy - target.vy),
    ])


def _disturbance_window(context, player):
    # Optional simulator metadata, never inferred from a student's agent.
    disturbance = getattr(context.models[player], "disturbance", None)
    if disturbance is None:
        return None
    start = float(disturbance.start_time)
    return start, start + float(disturbance.duration)


def _collision_times(context, player):
    return [float(report.at_time) for report in context.collision_reports if player in report.players]


def _decorate_time_axis(ax, context, player):
    window = _disturbance_window(context, player)
    if window:
        ax.axvspan(*window, color="darkorange", alpha=0.2, label="External disturbance")
    for index, t in enumerate(_collision_times(context, player)):
        ax.axvline(t, color="crimson", linestyle=":", label="Collision" if index == 0 else None)
    ax.grid(alpha=0.2)
    ax.set_xlabel("Simulation time [s]")


def plot_diagnostics(context, player, fig):
    """Goal tolerances, applied controls, fuel and callback timing after a run."""
    log, goal = context.log[player], context.missions[player]
    times = np.asarray(log.states.timestamps, dtype=float)
    states = log.states.values
    errors = np.asarray([goal_errors(state, goal) for state in states])
    axes = fig.subplots(3, 2)
    for column, (label, tol) in enumerate((("Position error [m]", goal.pos_tol),
                                          ("Heading error [rad]", goal.dir_tol),
                                          ("Velocity error [m/s]", goal.vel_tol))):
        ax = axes.flat[column]
        ax.plot(times, errors[:, column], label=label)
        ax.axhline(tol, color="darkorange", linestyle="--", label="Goal tolerance")
        ax.set_title(label)
    commands = log.commands
    control_ax = axes.flat[3]
    for field in ("thrust", "ddelta"):
        if commands.values and hasattr(commands.values[0], field):
            unit = "rad/s" if field == "ddelta" else "N"
            control_ax.plot(np.asarray(commands.timestamps, dtype=float),
                            [getattr(command, field) for command in commands.values], label=f"{field} [{unit}]")
    control_ax.set_title("Applied commands")
    fuel_ax = axes.flat[4]
    model = context.models[player]
    params = model.model_params
    if hasattr(states[0], "m") and hasattr(params, "m_v"):
        fuel_ax.plot(times, [state.m - params.m_v for state in states], label="Fuel remaining [kg]")
        fuel_ax.axhline(0, color="crimson", linestyle="--", label="Dry mass")
        fuel_ax.set_title("Fuel remaining [kg]")
    else:
        fuel_ax.plot(times, [np.hypot(state.vx, state.vy) for state in states], label="Speed [m/s]")
        fuel_ax.set_title("Speed [m/s]")
    timing_ax = axes.flat[5]
    timing_ax.plot(np.asarray(log.info.timestamps, dtype=float),
                   np.asarray(log.info.values, dtype=float) * 1000, label="get_commands [ms]")
    timing_ax.set_title("Command computation time [ms]")
    for ax in axes.flat:
        _decorate_time_axis(ax, context, player)
        ax.legend(fontsize=8)
    fig.suptitle(f"{player}: simulation diagnostics (velocity error uses body-frame components)")
    fig.tight_layout()


def _scene(context):
    """Use simulator geometry/colors and derive limits from the actual scenario."""
    renderer = SimRenderer(context, figsize=(12, 8))
    ax = renderer.commonroad_renderer.ax
    with renderer.plot_arena(ax=ax):
        pass
    # All three dock segments are drawn solid, so their geometry is unambiguous.
    from shapely.plotting import plot_polygon

    for goal in context.missions.values():
        if isinstance(goal, DockingTarget):
            plot_polygon(goal.get_plottable_geometry(), ax=ax, add_points=False,
                         facecolor="blue", edgecolor="blue", alpha=1.0, zorder=4)
    # YAML map boundaries are closed LineStrings among the static obstacles.
    # Moving obstacles must never determine the camera extent.
    boundaries = [obstacle.shape for obstacle in context.dg_scenario.static_obstacles
                  if obstacle.shape.geom_type in ("LineString", "LinearRing") and obstacle.shape.is_closed]
    if boundaries:
        bounds = np.asarray([boundary.bounds for boundary in boundaries])
        margin = 0.0
    else:
        # Without a boundary, frame the static scene and mission players only.
        bounds = [obstacle.shape.bounds for obstacle in context.dg_scenario.static_obstacles]
        for player, goal in context.missions.items():
            bounds.append(goal.get_plottable_geometry().bounds)
            xy = np.asarray([(state.x, state.y) for state in context.log[player].states.values])
            if len(xy):
                bounds.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
        bounds = np.asarray(bounds)
        margin = 1.0
    ax.set(xlim=(bounds[:, 0].min() - margin, bounds[:, 2].max() + margin),
           ylim=(bounds[:, 1].min() - margin, bounds[:, 3].max() + margin),
           xlabel="x [m]", ylabel="y [m]")
    ax.set_autoscale_on(False)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    return renderer, ax


def create_diagnostic_animation(file_path, context, fps=10):
    """Live goal/control status; the spaceship turns red during a disturbance."""
    renderer, ax = _scene(context)
    fig = ax.figure
    bodies, actions = {}, {}
    status = ax.text(1.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
                     family="monospace")
    disturbance_banner = ax.text(
        0.5, 0.98, "EXTERNAL DISTURBANCE", transform=ax.transAxes,
        ha="center", va="top", color="white", weight="bold",
        bbox=dict(facecolor="crimson", edgecolor="none", pad=5), visible=False,
    )
    fig.subplots_adjust(right=0.66)
    start = float(context.log.get_init_time())
    end = max(float(context.log[player].states.get_end()) for player in context.missions)
    frames = np.linspace(start, end, max(2, int(np.ceil((end - start) * fps)) + 1))
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    try:
        with writer.saving(fig, str(file_path), dpi=100):
            for t in frames:
                lines = [f"Simulation time: {t:.1f} s"]
                any_disturbance_active = False
                for player, log in context.log.items():
                    # Dynamic obstacles may stop logging before the spaceship does.
                    sample_t = min(max(t, float(log.states.get_start())), float(log.states.get_end()))
                    entry = log.at_interp(sample_t)
                    bodies[player], actions[player] = renderer.plot_player(
                        ax=ax, player_name=player, state=entry.state, command=entry.commands,
                        lights_colors=None, model_poly=bodies.get(player),
                        lights_patches=actions.get(player), alpha=0.8)
                    if player not in context.missions:
                        continue
                    state, goal = entry.state, context.missions[player]
                    window = _disturbance_window(context, player)
                    disturbance_active = window is not None and window[0] <= t < window[1]
                    any_disturbance_active |= disturbance_active
                    body_color = "crimson" if disturbance_active else context.models[player].model_geometry.color
                    bodies[player][0].set_facecolor(body_color)
                    bodies[player][0].set_edgecolor(body_color)
                    errors = goal_errors(state, goal)
                    lines += [f"\n{player}", "Goal errors / tolerances:"]
                    for label, value, tol, unit in zip(("pos", "heading", "vel"), errors,
                                                       (goal.pos_tol, goal.dir_tol, goal.vel_tol),
                                                       ("m", "rad", "m/s")):
                        lines.append(f" {label}: {value:.3f}/{tol:.3f} {unit}")
                    reached = goal.is_fulfilled(state, at=sample_t)
                    collided = any(ct <= t for ct in _collision_times(context, player))
                    lines += [f"Goal satisfied: {bool(reached)}", f"Collision: {collided}",
                              f"Speed: {np.hypot(state.vx, state.vy):.2f} m/s"]
                    for field in ("thrust", "ddelta"):
                        if hasattr(entry.commands, field):
                            unit = "rad/s" if field == "ddelta" else "N"
                            lines.append(f"{field}: {getattr(entry.commands, field):.2f} {unit}")
                    params = context.models[player].model_params
                    if hasattr(state, "m") and hasattr(params, "m_v"):
                        lines.append(f"Fuel: {state.m - params.m_v:.3f} kg")
                    lines.append(f"External disturbance: {'ON' if disturbance_active else 'OFF'}")
                status.set_text("\n".join(lines))
                disturbance_banner.set_visible(any_disturbance_active)
                writer.grab_frame()
    finally:
        plt.close(fig)
