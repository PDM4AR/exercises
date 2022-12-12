from typing import Tuple, List
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons.sim.simulator_animation import create_animation
from reprep import MIME_MP4, Report

from pdm4ar.exercises_def import Exercise
from pdm4ar.exercises_def.ex08.perf_metrics import ex08_metrics, Ex08Metrics
from pdm4ar.exercises_def.ex08.sim_context import get_sim_context_dynamic, get_sim_context_static


def ex08_evaluation(sim_context:SimContext, ex_out=None) -> Tuple[Ex08Metrics, Report]:
    r = Report("Final22-" + sim_context.description)

    sim = Simulator()
    sim.run(sim_context)
    report = _ex08_vis(sim_context=sim_context)
    episode_eval = ex08_metrics(sim_context)
    r.text("EpisodeEvaluation", remove_escapes(debug_print(episode_eval)))
    r.add_child(report)
    return episode_eval, r


def ex08_performance_aggregator(ex_out:List[Ex08Metrics]) -> Ex08Metrics:
    # todo 
    pass


def _ex08_vis(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_MP4) as fn:
        create_animation(file_path=fn,
                         sim_context=sim_context,
                         figsize=(16, 16),
                         dt=50, dpi=120,
                         plot_limits=None)

    return r


def get_exercise8():
    seed = 98
    test_values:List[SimContext] = [
        get_sim_context_static(seed),
        get_sim_context_dynamic(seed)
        ]

    return Exercise[SimContext, None](
            desc="Final '22 planning course exercise.",
            evaluation_fun=ex08_evaluation,
            perf_aggregator=lambda x: ex08_performance_aggregator(x),
            test_values=test_values,
            expected_results=[None, ]*len(test_values),
            test_case_timeout=60 * 20,  # For debugging, increase value if your report generation is slow!
    )
