import pprint
from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons.sim.simulator_animation import create_animation
from reprep import MIME_MP4, Report

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex10.perf_metrics import ex10_metrics
from pdm4ar.exercises_def.ex10.utils_config import sim_context_from_yaml


def ex10_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[float, Report]:
    r = Report("Final23-" + sim_context.description)
    # run simulation
    sim = Simulator()
    sim.run(sim_context)
    # visualisation
    report = _ex10_vis(sim_context=sim_context)
    # compute metrics
    avg_player_metrics, players_metrics = ex10_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    score_str = f"{score:.2f}\n" + pprint.pformat(avg_player_metrics)
    r.text("OverallScore: ", score_str)
    for pm in players_metrics:
        r.text(f"EpisodeEvaluation-{pm.player_name}", pprint.pformat(pm))
    r.add_child(report)
    return score, r


def ex10_performance_aggregator(ex_out: List[float]) -> float:
    return np.average(ex_out)


def _ex10_vis(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(file_path=fn,
                         sim_context=sim_context,
                         figsize=(16, 16),
                         dt=50,
                         dpi=120,
                         plot_limits=[[-12, 12], [-12, 12]] # make sure this is aligned with boundaries
                         )

    return r


def get_exercise10():
    config_dir = Path(__file__).parent
    name_1, name_2 = "config_1.yaml", "config_2.yaml"
    test_values: List[SimContext] = [
        sim_context_from_yaml(str(config_dir / name_1)),
        sim_context_from_yaml(str(config_dir / name_2))
    ]

    return Exercise[SimContext, None](
            desc="Final'23- planning course exercise.",
            evaluation_fun=ex10_evaluation,
            perf_aggregator=lambda x: ex10_performance_aggregator(x),
            test_values=test_values,
            expected_results=[None, ] * len(test_values),
            test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
