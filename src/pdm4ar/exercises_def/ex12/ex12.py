from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons.sim.simulator_animation import create_animation
from reprep import MIME_MP4, Report
import pprint
from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex12.perf_metrics import (
    PlayerMetrics,
    ex12_metrics,
    HighwayFinalPerformance,
    get_task_performance,
)
from pdm4ar.exercises_def.ex12.sim_context import get_sim_contexts
from .utils import get_collision_reports


def ex12_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[PlayerMetrics, Report]:
    r = Report("Final24-" + sim_context.description)
    # run simulation
    sim = Simulator()
    sim.run(sim_context)
    # visualisation
    report = _ex12_vis(sim_context=sim_context)
    # compute metrics
    player_metrics = ex12_metrics(sim_context)
    # report evaluation
    score: float = player_metrics.reduce_to_score()
    score_str = f"{score:.2f}\n" + str(player_metrics)
    r.text("Evaluation: ", text=pprint.pformat(player_metrics))
    r.add_child(report)
    if player_metrics.collided:
        collision_report = get_collision_reports(sim_context, skip_collision_viz=True)
        r.add_child(collision_report)
    return player_metrics, r


def ex12_performance_aggregator(ex_out: List[PlayerMetrics]) -> HighwayFinalPerformance:
    metrics_by_level = {1: [], 2: [], 3: []}
    for metrics in ex_out:
        if metrics.task_level == 1:
            metrics_by_level[1].append(metrics)
        elif metrics.task_level == 2:
            metrics_by_level[2].append(metrics)
        elif metrics.task_level == 3:
            metrics_by_level[3].append(metrics)
        else:
            # this line should not be reached
            pass
    task_performances = {
        1: get_task_performance(metrics_by_level[1]),
        2: get_task_performance(metrics_by_level[2]),
        3: get_task_performance(metrics_by_level[3]),
    }
    final_score = (
        0.4 * task_performances[1].avg_score
        + 0.4 * task_performances[2].avg_score
        + 0.2 * task_performances[3].avg_score
    )
    return HighwayFinalPerformance(
        final_score=final_score,
        task_performances=task_performances,
    )


def _ex12_vis(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 16), dt=50, dpi=120)

    return r


def load_config_ex12(file_path: Path) -> Mapping:
    with open(str(file_path)) as f:
        config_dict = yaml.safe_load(f)
    return fd(config_dict)


def get_exercise12():
    config_dir = Path(__file__).parent
    scenarios_dir = str(config_dir)
    config_list = ["config_1.yaml", "config_2.yaml", "config_3.yaml"]
    test_values: List[SimContext] = []
    for config_name in config_list:
        config_dict = load_config_ex12(config_dir / config_name)
        test_values += get_sim_contexts(config_dict, scenarios_dir)

    return Exercise[SimContext, None](
        desc="Final '24 planning course exercise.",
        evaluation_fun=ex12_evaluation,
        perf_aggregator=lambda x: ex12_performance_aggregator(x),
        test_values=test_values,
        expected_results=[
            None,
        ]
        * len(test_values),
        test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
