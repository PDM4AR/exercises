import pprint
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex14.perf_metrics import ex14_metrics
from pdm4ar.exercises_def.ex14.utils_config import load_config, sim_context_from_config
from reprep import MIME_MP4, Report


def ex14_evaluation(sim_config: Mapping[str, Any], ex_out=None) -> Tuple[float, Report]:
    sim_context: SimContext = sim_context_from_config(sim_config)
    r = Report("Final25-" + sim_context.description)
    # run simulation
    sim = Simulator()
    sim.run(sim_context)
    # visualisation
    report = _ex14_vis(sim_context=sim_context)
    # compute metrics
    avg_player_metrics, players_metrics = ex14_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    score_str = f"{score:.2f}\n" + pprint.pformat(avg_player_metrics)
    r.text("OverallScore: ", score_str)
    for pm in players_metrics:
        r.text(f"EpisodeEvaluation-{pm.player_name}", pprint.pformat(pm))
    r.add_child(report)
    return score, r


def ex14_performance_aggregator(ex_out: List[float]) -> float:
    return np.average(ex_out)


def _ex14_vis(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(
            file_path=fn,
            sim_context=sim_context,
            figsize=(16, 16),
            dt=50,
            dpi=60,
            plot_limits=[[-12, 12], [-12, 12]],  # make sure this is aligned with boundaries
        )

    return r


def get_exercise14():
    config_dir = Path(__file__).parent
    config_files = ["config_test.yaml"]
    test_values: List[Mapping[str, Any]] = [load_config(str(config_dir / config_file)) for config_file in config_files]

    return Exercise[SimContext, None](
        desc="Final'25- planning course exercise.",
        evaluation_fun=ex14_evaluation,
        perf_aggregator=lambda x: ex14_performance_aggregator(x),
        test_values=test_values,
        expected_results=[
            None,
        ]
        * len(test_values),
        test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
