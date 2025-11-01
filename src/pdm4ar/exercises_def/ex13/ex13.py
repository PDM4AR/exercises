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
from pdm4ar.exercises_def.ex13.get_config import get_config


def ex13_evaluation(sim_context: SimContext, ex_out=None) -> Tuple[Tuple[str, float], Report]:
    r = Report("Final25-" + sim_context.description)
    # run simulation
    run_simulation(sim_context)
    # visualisation
    report = _ex13_vis(sim_context=sim_context)
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


def _ex13_vis(sim_context: SimContext) -> Report:
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
