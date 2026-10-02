import pprint
from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.log_visualisation import plot_player_log
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.utils import run_simulation
from reprep import MIME_MP4, Report
from collections import defaultdict

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex13.perf_metrics import ex13_metrics
from pdm4ar.exercises_def.ex13.utils_config import sim_context_from_yaml
from pdm4ar.exercises_def.ex13.disturbed_spaceship import DisturbedSpaceshipModel
from pdm4ar.exercises_def.ex13.get_config import get_config


def ex13_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[Tuple[str, float], Report]:
    r = Report("Final25-" + sim_context.description)
    agents = dict(sim_context.players)
    # run simulation
    run_simulation(sim_context)
    # visualisation
    report = _ex13_vis(sim_context=sim_context, agents=agents)
    # compute metrics
    avg_player_metrics, _ = ex13_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    r.text(f"EpisodeEvaluation:", pprint.pformat(avg_player_metrics))
    score_str = f"{score:.2f}"
    r.text("OverallScore: ", score_str)
    r.add_child(report)
    return (sim_context.description, score), r


def ex13_performance_aggregator(ex_out: List[Tuple[str, float]]) -> Tuple[str, float]:
    # Compute the average score for each scenario (string key) in the list of results (ex_out).
    score_dict = defaultdict(list)
    for k, v in ex_out:
        score_dict[k].append(v)
    score_per_scenario = {k: float(np.mean(vs)) for k, vs in score_dict.items()}
    scores = {"Average": float(np.mean(list(score_per_scenario.values()))), "Per Scenario": score_per_scenario}
    return scores


def _plot_controller_events(fig, agent, model) -> None:
    """Plot plans, tracking errors, commands, and the actual disturbance window."""
    fig.clear()
    ax_traj, ax_error, ax_commands = fig.subplots(1, 3)

    times = np.asarray(agent.command_times, dtype=float)
    states = agent.actual_trajectory
    sample_count = min(len(times), len(states))
    times = times[:sample_count]
    states = states[:sample_count]

    for index, plan in enumerate(agent.planned_trajectories):
        plan_values = list(plan._values)
        ax_traj.plot(
            [state.x for state in plan_values],
            [state.y for state in plan_values],
            linestyle="--",
            alpha=0.65,
            label="Initial plan" if index == 0 else f"Replan {index}",
        )

    if states:
        actual_x = np.asarray([state.x for state in states])
        actual_y = np.asarray([state.y for state in states])
        ax_traj.plot(actual_x, actual_y, color="tab:blue", linewidth=2.0, label="Actual trajectory")
    else:
        actual_x = np.asarray([])
        actual_y = np.asarray([])

    disturbance_window = None
    if isinstance(model, DisturbedSpaceshipModel):
        start = float(model.disturbance.start_time)
        end = start + float(model.disturbance.duration)
        disturbance_window = (start, end)
        if sample_count:
            active = (times >= start) & (times < end)
            disturbed_x = np.where(active, actual_x, np.nan)
            disturbed_y = np.where(active, actual_y, np.nan)
            ax_traj.plot(
                disturbed_x, disturbed_y, color="tab:orange", linewidth=5.0,
                label=f"Disturbance active ({start:.1f}-{end:.1f} s)",
            )

    for index, replan_time in enumerate(agent.replanning_times, start=1):
        if sample_count:
            state_index = int(np.argmin(np.abs(times - replan_time)))
            ax_traj.scatter(
                actual_x[state_index], actual_y[state_index], marker="X", s=110,
                color="tab:red", edgecolor="black", zorder=10,
                label="Replanning" if index == 1 else None,
            )
            ax_traj.annotate(
                f"R{index}  t={replan_time:.1f}s",
                (actual_x[state_index], actual_y[state_index]),
                xytext=(7, 7), textcoords="offset points", fontsize=8,
            )

    error_count = min(len(times), len(agent.norm2_errors), len(agent.max_errors))
    error_times = times[:error_count]
    ax_error.plot(error_times, agent.norm2_errors[:error_count], label="State error norm")
    ax_error.plot(error_times, agent.max_errors[:error_count], label="Maximum component error", alpha=0.8)

    command_count = min(len(times), len(agent.command_components))
    if command_count:
        commands = np.asarray(agent.command_components[:command_count])
        ax_commands.plot(times[:command_count], commands[:, 0], label="Thrust [N]")
        ax_commands.plot(times[:command_count], commands[:, 1], label="Nozzle rate [rad/s]")

    for axis in (ax_error, ax_commands):
        if disturbance_window is not None:
            axis.axvspan(
                disturbance_window[0], disturbance_window[1], color="tab:orange", alpha=0.22,
                label="Disturbance active" if axis is ax_error else None,
            )
        for index, replan_time in enumerate(agent.replanning_times, start=1):
            axis.axvline(
                replan_time, color="tab:red", linestyle="--", linewidth=1.4,
                label="Replanning" if axis is ax_error and index == 1 else None,
            )

    ax_traj.set(title="Trajectory and replanning events", xlabel="x [m]", ylabel="y [m]")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_error.set(title="Tracking error and events", xlabel="Simulation time [s]", ylabel="Error")
    ax_commands.set(title="Actuation and events", xlabel="Simulation time [s]", ylabel="Command")
    for axis in (ax_traj, ax_error, ax_commands):
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()


def _ex13_vis(sim_context: SimContext, agents=None) -> Report:
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
                with sub.plot(f"{pn}-controller-events", figsize=(24, 8)) as pylab:
                    _plot_controller_events(pylab.gcf(), agent, sim_context.models[pn])
    return r


def load_config_ex13(file_path: Path) -> Mapping:
    with open(str(file_path)) as f:
        config_dict = yaml.safe_load(f)
    return fd(config_dict)


def get_exercise13():
    config_dir = Path(__file__).parent
    configs = get_config()

    test_values: List[SimContext] = []
    for c in configs:
        config_file = config_dir / c
        sim_context = sim_context_from_yaml(str(config_file))
        test_values.append(sim_context)

    return Exercise[SimContext, None](
        desc="PDM4ARSpaceship(ex13)",
        evaluation_fun=ex13_evaluation,
        perf_aggregator=ex13_performance_aggregator,
        test_values=test_values,
        expected_results=[
            None,
        ]
        * len(test_values),
        test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
