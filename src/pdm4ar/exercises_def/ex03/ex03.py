from itertools import product
from typing import Tuple, Any

import osmnx as ox
from time import process_time
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF, Node
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def import Exercise, NodeColors, EdgeColors, ExIn, Ex02PerformanceResult
from pdm4ar.exercises.ex03 import informed_graph_search_algo, compute_path_cost
from pdm4ar.exercises_def.ex02 import Ex02PerformanceResult, ex2_perf_aggregator, str_from_path
from pdm4ar.exercises_def.ex03.data import ex3_get_expected_results, get_test_informed_gsproblem, \
    InformedGraphSearchProblem


class TestValueEx3(ExIn, Tuple[InformedGraphSearchProblem, str]):
    def str_id(self) -> str:
        return str(self[1])


def ex3_evaluation(ex_in: TestValueEx3, ex_out=None) -> Tuple[Ex02PerformanceResult, Report]:
    # ex properties
    prob, algo_name = ex_in
    wG = prob.graph
    test_queries = prob.queries
    ec = [EdgeColors.default for uv in wG._G.edges]
    # init report
    r = Report(f"Exercise3-{algo_name}-{prob.graph_id}")
    # draw graph
    figsize = None
    rfig = r.figure(cols=1)
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=figsize) as _:
        ax = plt.gca()
        ox.plot_graph(
                wG._G, ax=ax, node_color=NodeColors.default, edge_color=EdgeColors.default, node_edgecolor="k",
                show=False
        )

    # run algo
    r.section(f"{algo_name}")
    solve_times = []
    accuracy = []
    for i, query in enumerate(test_queries):
        nc = [
            NodeColors.start if n == query[0] else (NodeColors.goal if n == query[1] else NodeColors.default)
            for n in wG._G
        ]
        # Ground truth
        msg = f"Start: {query[0]},\tGoal: {query[1]}\n"
        search_algo = informed_graph_search_algo[algo_name](wG)
        rfig = r.figure(cols=2)
        # Your algo
        start = process_time()
        path = search_algo.path(query[0], query[1])
        solve_time = process_time() - start
        if path:
            path_str = str_from_path(path)
            path_cost = compute_path_cost(wG, path)
            with rfig.plot(nid=f"YourPath{i}-{algo_name}", mime=MIME_PDF, figsize=figsize) as _:
                ax = plt.gca()
                ox.plot_graph(wG._G, ax=ax, node_color=nc, node_edgecolor="k", edge_color=ec, show=False, close=False)
                ox.plot_graph_route(wG._G, route=path, ax=ax, orig_dest_size=0, route_linewidth=1, show=False,
                                    close=False)
        else:
            path_str = "Your algo did not find any path."
            path_cost = float("inf")
            path = []
        # ground truths
        gt_path = ex_out[i]
        # compare to ground truth only for admissible heuristic
        if gt_path is not None:
            # Compute gt path cost
            gt_path_cost = compute_path_cost(wG, gt_path)
            gt_path_str = str_from_path(gt_path)
            # Plot ground truth
            with rfig.plot(nid=f"GroundTruth{i}-{algo_name}", mime=MIME_PDF, figsize=figsize) as _:
                ax = plt.gca()
                ox.plot_graph(wG._G, ax=ax, node_color=nc, node_edgecolor="k", edge_color=ec, show=False, close=False)
                ox.plot_graph_route(wG._G, route=gt_path, ax=ax, orig_dest_size=0, route_linewidth=1, show=False,
                                    close=False)
            # Compare your algo to ground truth
            if gt_path_cost == path_cost:
                accuracy.append(1.)
                msg += "Student solution : CORRECT\n"
            else:
                accuracy.append(0.)
                msg += "Student solution : WRONG\n"
            solve_times.append(solve_time)
        else:
            gt_path_str = "Solution not given"

        # output path to report
        msg += f"Ground truth path: {gt_path_str}\n"
        msg += f"Ground truth path cost: {gt_path_cost:.2f}\n"

        msg += f"Your path: {path_str}\n"
        msg += f"Your path cost:\t{path_cost:.2f}\n"

        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))

    # aggregate performance of each query
    query_perf = list(map(Ex02PerformanceResult, accuracy, solve_times))
    perf = ex2_perf_aggregator(query_perf)
    return perf, r


def get_exercise3() -> Exercise:
    test_wgraphs = get_test_informed_gsproblem(n_queries=1, n_seed=4)
    expected_results = ex3_get_expected_results()
    test_values = list()
    for ab in product(test_wgraphs, informed_graph_search_algo):
        test_values.append(TestValueEx3(ab))

    return Exercise[TestValueEx3, Any](
            desc='This exercise is about graph search',
            evaluation_fun=ex3_evaluation,
            perf_aggregator=ex2_perf_aggregator,
            test_values=test_values,
            expected_results=expected_results,
    )
