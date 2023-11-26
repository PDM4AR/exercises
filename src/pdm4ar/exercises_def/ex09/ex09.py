from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd, PlayerName
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.utils import run_simulation
from reprep import MIME_MP4, Report

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex09.utils_config import sim_context_from_yaml


def ex09_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[float, Report]:
    r = Report("Final23-" + sim_context.description)
    # run simulation
    run_simulation(sim_context)
    # visualisation
    report = _ex09_vis(sim_context=sim_context)
    # TODO metrics
    score = 1
    # # compute metrics
    # avg_player_metrics, players_metrics = ex09_metrics(sim_context)
    # # report evaluation
    # score: float = avg_player_metrics.reduce_to_score()
    # score_str = f"{score:.2f}\n" + str(avg_player_metrics)
    # r.text("OverallScore: ", score_str)
    # for pm in players_metrics:
    #     r.text(f"EpisodeEvaluation-{pm.player_name}", str(pm))
    r.add_child(report)
    return score, r


def ex09_performance_aggregator(ex_out: List[float]) -> float:
    return np.average(ex_out)


def _ex09_vis(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(file_path=fn,
                         sim_context=sim_context,
                         figsize=(16, 16),
                         dt=50,
                         dpi=120,
                         plot_limits=PlayerName("PDM4ARocket")
                         )

    return r


def load_config_ex08(file_path: Path) -> Mapping:
    with open(str(file_path)) as f:
        config_dict = yaml.safe_load(f)
    return fd(config_dict)


def get_exercise09():
    config_dir = Path(__file__).parent
    configs = ["config_planet.yaml", "config_satellites.yaml", ]  # "config_mov_target.yaml"]

    test_values: List[SimContext] = []
    for c in configs:
        config_file = config_dir / c
        sim_context = sim_context_from_yaml(str(config_file))
        test_values.append(sim_context)

    return Exercise[SimContext, None](
            desc="PDM4ARocket(ex09)",
            evaluation_fun=ex09_evaluation,
            perf_aggregator=lambda x: ex09_performance_aggregator(x),
            test_values=test_values,
            expected_results=[None, ] * len(test_values),
            test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
