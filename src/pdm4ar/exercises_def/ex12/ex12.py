from pathlib import Path
from typing import Tuple, List, Mapping

import numpy as np
import yaml
from dg_commons import fd
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons.sim.simulator_animation import create_animation
from reprep import MIME_MP4, Report
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex12.perf_metrics import ex12_metrics
from pdm4ar.exercises_def.ex12.sim_context import get_sim_contexts


def ex12_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[float, Report]:
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
    r.text("Score: ", score_str)
    r.add_child(report)
    return score, r


def ex12_performance_aggregator(ex_out: List[float]) -> float:
    return np.average(ex_out)


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
    name_1 = "config_1.yaml"
    config_dict_1 = load_config_ex12(config_dir / name_1)
    # config_dict_2 = load_config_ex12(config_dir / name_2)
    test_values: List[SimContext] = get_sim_contexts(config_dict_1)

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
