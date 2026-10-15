import pprint
from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.log_visualisation import plot_player_log
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from reprep import MIME_MP4, Report
from collections import defaultdict

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex15.perf_metrics import ex15_metrics
from pdm4ar.exercises_def.ex15.utils_config import sim_context_from_yaml
from pdm4ar.exercises_def.ex15.disturbed_spaceship import DisturbedSpaceshipModel
from pdm4ar.exercises_def.ex15.get_config import get_config


def run_simulation(sim_context: SimContext) -> None:
    """Run an episode without inheriting mutable state from previous episodes."""
    simulator = Simulator()
    simulator.last_commands = {}
    simulator.simlogger = {}
    simulator.disabled_players = []
    simulator.run(sim_context)


def ex15_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[Tuple[str, float], Report]:
    r = Report("Exercise15-" + sim_context.description)
    agents = dict(sim_context.players)
    # run simulation
    run_simulation(sim_context)
    # visualisation
    report = _ex15_vis(sim_context=sim_context, agents=agents)
    # compute metrics
    avg_player_metrics, _ = ex15_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    r.text(f"EpisodeEvaluation:", pprint.pformat(avg_player_metrics))
    score_str = f"{score:.2f}"
    r.text("OverallScore: ", score_str)
    r.add_child(report)
    return (sim_context.description, score), r


def ex15_performance_aggregator(ex_out: List[Tuple[str, float]]) -> Tuple[str, float]:
    # Compute the average score for each scenario (string key) in the list of results (ex_out).
    score_dict = defaultdict(list)
    for k, v in ex_out:
        score_dict[k].append(v)
    score_per_scenario = {k: float(np.mean(vs)) for k, vs in score_dict.items()}
    scores = {"Average": float(np.mean(list(score_per_scenario.values()))), "Per Scenario": score_per_scenario}
    return scores


def _plot_controller_events(fig, agent, model, sim_context=None) -> None:
    """Show the executed route, successful replans, docking handoff and controls."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from shapely.plotting import plot_polygon

    fig.clear()
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), hspace=0.35, wspace=0.25)
    ax_traj = fig.add_subplot(grid[:, 0])
    ax_error = fig.add_subplot(grid[0, 1])
    ax_commands = fig.add_subplot(grid[1, 1])
    times = np.asarray(agent.command_times, dtype=float)
    count = min(len(times), len(agent.actual_trajectory))
    if not count:
        return
    times = times[:count]
    states = agent.actual_trajectory[:count]
    xy = np.array([[state.x, state.y] for state in states])
    end_time = times[-1]
    route_color, event_color, noise_color = "#177f99", "#bd3b58", "#ed8b36"

    # Give the paths their physical context, including obstacle motion up to arrival.
    for name, planet in agent.planets.items():
        ax_traj.add_patch(Circle(planet.center, planet.radius, facecolor="#c4cddd", edgecolor="#8191a8", zorder=2))
        ax_traj.text(*planet.center, name, ha="center", va="center", fontsize=9, color="#344459")
    if sim_context is not None:
        for name, asteroid in agent.asteroids.items():
            log = sim_context.log[name].states
            positions = np.array([[state.x, state.y] for time, state in zip(log.timestamps, log.values) if float(time) <= end_time])
            if len(positions):
                ax_traj.plot(*positions.T, "--", color="#8865ac", linewidth=1.4, alpha=0.8)
                for point, alpha in ((positions[0], 0.18), (positions[-1], 0.5)):
                    ax_traj.add_patch(Circle(point, asteroid.radius, color="#8865ac", alpha=alpha))
                ax_traj.annotate(name, positions[0], xytext=(0, 12), textcoords="offset points", fontsize=8, color="#694c89")
                if len(positions) > 1:
                    ax_traj.annotate("", positions[-1], positions[len(positions)//2], arrowprops=dict(arrowstyle="->", color="#8865ac"))
    if hasattr(agent.goal, "target"):
        if hasattr(agent.goal, "get_landing_constraint_points"):
            plot_polygon(agent.goal.get_plottable_geometry(), ax=ax_traj, add_points=False, color="#315b99", alpha=0.8, zorder=4)
        target = agent.goal.target
        ax_traj.scatter(target.x, target.y, marker="*", s=150, color="#e5b04b", edgecolors="#98721e", zorder=8)
    border = agent.map_borders
    ax_traj.set(xlim=(border["xmin"]-0.5, border["xmax"]+0.5), ylim=(border["ymin"]-0.5, border["ymax"]+0.5))

    for index, plan in enumerate(agent.planned_trajectories):
        values = list(plan.values)
        ax_traj.plot([state.x for state in values], [state.y for state in values], linestyle=":" if index == 0 else "--", color="#a7b4c5", linewidth=1, alpha=0.8, zorder=1)
    ax_traj.plot(*xy.T, color=route_color, linewidth=2.5, zorder=5)
    ax_traj.scatter(*xy[0], s=65, color="#249d7b", edgecolors="white", zorder=8)

    window = None
    if isinstance(model, DisturbedSpaceshipModel):
        start = float(model.disturbance.start_time)
        end = start + float(model.disturbance.duration)
        window = (start, end)
        active = (times >= start) & (times <= end)
        ax_traj.plot(np.where(active, xy[:, 0], np.nan), np.where(active, xy[:, 1], np.nan), color=noise_color, linewidth=5, zorder=6)

    # Entry planning is a stage transition; distinguish it from error-triggered replanning.
    events = list(agent.replanning_times)
    has_handoff = agent.dock_stage == "entry" and bool(events)
    for index, event_time in enumerate(events):
        handoff = has_handoff and index == len(events)-1
        label = "D" if handoff else f"R{index+1}"
        color = "#287b6d" if handoff else event_color
        marker = "D" if handoff else "X"
        position = xy[int(np.argmin(np.abs(times-event_time)))]
        ax_traj.scatter(*position, marker=marker, s=100, color=color, edgecolors="white", linewidth=0.8, zorder=10)
        offset = (12, 30 if index % 2 == 0 else -35)
        ax_traj.annotate(f"{label} · {event_time:.1f} s", position, xytext=offset, textcoords="offset points", fontsize=9, color=color,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95),
                         arrowprops=dict(arrowstyle="-", color=color, lw=0.8), zorder=11)
        for axis in (ax_error, ax_commands):
            axis.axvline(event_time, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax_error.text(event_time, 0.95 if index % 2 == 0 else 0.86, label, transform=ax_error.get_xaxis_transform(), ha="center", va="top", color=color, fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))

    n = min(count, len(agent.norm2_errors), len(agent.max_errors))
    ax_error.plot(times[:n], agent.norm2_errors[:n], color=route_color, label="State error norm", linewidth=1.5)
    ax_error.plot(times[:n], agent.max_errors[:n], color="#8b9aaf", label="Largest component", linewidth=1)
    n = min(count, len(agent.command_components))
    if n:
        commands = np.asarray(agent.command_components[:n])
        ax_commands.plot(times[:n], commands[:, 0], color=route_color, label="Thrust [N]")
        ax_commands.plot(times[:n], commands[:, 1], color="#8865ac", label="Nozzle rate [rad/s]")
    for axis in (ax_error, ax_commands):
        if window is not None:
            axis.axvspan(*window, color=noise_color, alpha=0.17, label="Disturbance")
        axis.set_xlim(0, end_time)
        axis.set_xlabel("Simulation time [s]")
        axis.legend(loc="upper right", fontsize=8, framealpha=0.95)
    handles = [Line2D([], [], color=route_color, lw=2.5, label="Executed route"),
               Line2D([], [], color="#a7b4c5", ls="--", label="Planned routes")]
    if window is not None:
        handles.append(Line2D([], [], color=noise_color, lw=4, label="Disturbance active"))
    if len(events) > int(has_handoff):
        handles.append(Line2D([], [], color=event_color, marker="X", ls="", label="R: successful replan"))
    if has_handoff:
        handles.append(Line2D([], [], color="#287b6d", marker="D", ls="", label="D: docking entry plan"))
    if agent.asteroids:
        handles.append(Line2D([], [], color="#8865ac", ls="--", label="Moving asteroids"))
    ax_traj.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.95)
    ax_traj.set(title="Route and planning events", xlabel="x [m]", ylabel="y [m]")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_error.set(title="Tracking error", ylabel="State error")
    ax_commands.set(title="Applied controls", ylabel="Command")
    for axis in (ax_traj, ax_error, ax_commands):
        axis.set_facecolor("#f7f9fc")
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    description = sim_context.description if sim_context is not None else "Spaceship controller"
    fig.suptitle(description.replace("_", " "), fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98)

def _ex15_vis(sim_context: SimContext, agents=None) -> Report:
    agents = dict(sim_context.players) if agents is None else agents
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(
            file_path=fn,
            sim_context=sim_context,
            figsize=(16, 16),
            dt=50,
            dpi=120,
            plot_limits=[[-12, 27], [-12, 12]],  # plot_limits = [horizontal, vertical]
        )
    # state/commands plots
    for pn in sim_context.log.keys():
        if "PDM4AR" not in pn:
            continue
        with r.subsection(f"Player-{pn}-log") as sub:
            with sub.plot(f"{pn}-log", figsize=(20, 15)) as pylab:
                plot_player_log(log=sim_context.log[pn], fig=pylab.gcf())
            agent = agents.get(pn)
            if agent is not None and hasattr(agent, "replanning_times"):
                with sub.plot(f"{pn}-controller-events", figsize=(17, 9)) as pylab:
                    _plot_controller_events(pylab.gcf(), agent, sim_context.models[pn], sim_context)
    return r


def load_config_ex15(file_path: Path) -> Mapping:
    with open(str(file_path)) as f:
        config_dict = yaml.safe_load(f)
    return fd(config_dict)


def get_exercise15():
    config_dir = Path(__file__).parent
    configs = get_config()

    test_values: List[SimContext] = []
    for c in configs:
        config_file = config_dir / c
        sim_context = sim_context_from_yaml(str(config_file))
        test_values.append(sim_context)

    return Exercise[SimContext, None](
        desc="Robust satellite docking (ex15)",
        evaluation_fun=ex15_evaluation,
        perf_aggregator=ex15_performance_aggregator,
        test_values=test_values,
        expected_results=[
            None,
        ]
        * len(test_values),
        test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
