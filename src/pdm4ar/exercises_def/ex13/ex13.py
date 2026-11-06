import pprint
from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
from matplotlib import pyplot as plt
import yaml
from dg_commons import fd
from dg_commons.sim.log_visualisation import plot_player_log
from dg_commons.sim.simulator import SimContext
from pdm4ar.exercises_def.ex13.visualization import (
    create_diagnostic_animation,
    plot_diagnostics,
    plot_trajectory_overview,
    record_tracking,
)
from dg_commons.sim.utils import run_simulation
from reprep import MIME_MP4, MIME_PNG, Report
from collections import defaultdict

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex13.perf_metrics import ex13_metrics
from pdm4ar.exercises_def.ex13.utils_config import sim_context_from_yaml
from pdm4ar.exercises_def.ex13.get_config import get_config


def ex13_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[Tuple[str, float], Report]:
    r = Report(sim_context.description)
    record_tracking(sim_context)
    run_simulation(sim_context)
    report = _ex13_vis(sim_context=sim_context)
    avg_player_metrics, _ = ex13_metrics(sim_context)
    score: float = avg_player_metrics.reduce_to_score()
    r.text("Summary", _ex13_summary(avg_player_metrics, score))
    r.text("Metrics", pprint.pformat(avg_player_metrics))
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


def _ex13_summary(m, score: float) -> str:
    """Outcome first, then how much each term of reduce_to_score costs."""
    outcome = "GOAL REACHED" if m.goal_success_rate >= 1 else "GOAL NOT REACHED"
    if m.collision_rate > 0:
        outcome += " - COLLISION"
    return "\n".join([
        f"{outcome}   |   score {score:.2f}",
        "",
        "Score = 1000 * (goal - collision) - penalties:",
        f"  computation time  {m.avg_computation_time:7.3f} s/step  -> -{m.avg_computation_time * 1e2:6.2f}",
        f"  distance to goal  {m.avg_distance2goal:7.2f} m       -> -{m.avg_distance2goal / 2 * 1e1:6.2f}",
        f"  distance flown    {m.avg_distance_travelled:7.2f} m       -> -{m.avg_distance_travelled / 5 * 1e1:6.2f}",
        f"  episode duration  {m.avg_episode_duration:7.2f} s       -> -{m.avg_episode_duration / 3 * 1e1:6.2f}",
        f"  actuation effort  {m.avg_actuation_effort:7.2f}         -> -{m.avg_actuation_effort * 1e1:6.2f}",
    ])


def _ex13_vis(sim_context: SimContext) -> Report:
    r = Report("Visualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Trajectory", MIME_PNG) as fn:
        fig = plot_trajectory_overview(sim_context)
        fig.savefig(fn, dpi=100)
        plt.close(fig)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_diagnostic_animation(fn, sim_context)
    # goal errors, commands, fuel, timing; then the raw state/commands log
    for pn in sim_context.log.keys():
        if "PDM4AR" not in pn:
            continue
        with r.subsection("Diagnostics") as sub:
            with sub.plot("diagnostics", figsize=(14, 13)) as pylab:
                plot_diagnostics(sim_context, pn, pylab.gcf())
            with sub.plot("log", figsize=(14, 10)) as pylab:
                plot_player_log(log=sim_context.log[pn], fig=pylab.gcf())
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
