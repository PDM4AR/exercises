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
from pdm4ar.exercises_def.ex08.perf_metrics import ex08_metrics
from pdm4ar.exercises_def.ex08.sim_context import get_sim_context


def ex09_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[float, Report]:
    r = Report("Final23-" + sim_context.description)
    # run simulation
    run_simulation(sim_context)
    # visualisation
    report = _ex08_vis(sim_context=sim_context)
    # compute metrics
    avg_player_metrics, players_metrics = ex08_metrics(sim_context)
    # report evaluation
    score: float = avg_player_metrics.reduce_to_score()
    score_str = f"{score:.2f}\n" + str(avg_player_metrics)
    r.text("OverallScore: ", score_str)
    for pm in players_metrics:
        r.text(f"EpisodeEvaluation-{pm.player_name}", str(pm))
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
                         dpi=120
                         )

    return r


def load_config_ex08(file_path: Path)->Mapping:
    with open(str(file_path)) as f:
        config_dict = yaml.safe_load(f)
    return fd(config_dict)


def get_exercise8():
    config_dir = Path(__file__).parent
    name_1, name_2 = "config_1.yaml", "config_2.yaml"
    config_dict_1 = load_config_ex08(config_dir / name_1)
    config_dict_2 = load_config_ex08(config_dir / name_2)
    test_values: List[SimContext] = [
        get_sim_context(config_dict_1, config_dict_1["seed"], config_name=name_1),
        get_sim_context(config_dict_2, config_dict_2["seed"], config_name=name_2)
    ]

    return Exercise[SimContext, None](
            desc="Final '22 planning course exercise.",
            evaluation_fun=ex08_evaluation,
            perf_aggregator=lambda x: ex08_performance_aggregator(x),
            test_values=test_values,
            expected_results=[None, ] * len(test_values),
            test_case_timeout=60 * 10,  # For debugging, increase value if your report generation is slow!
    )
