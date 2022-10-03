from itertools import product
from typing import Tuple, Any

import osmnx as ox
from time import process_time
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF
from zuper_commons.text import remove_escapes

from pdm4ar.exercises.ex03 import informed_graph_search_algo, compute_path_cost, Heuristic
from pdm4ar.exercises_def import Exercise, NodeColors, EdgeColors, ExIn
from pdm4ar.exercises_def.ex03.data import ex3_get_expected_results, get_test_informed_gsproblem, InformedGraphSearchProblem
from pdm4ar.exercises_def.ex02 import Ex02PerformanceResult, ex2_perf_aggregator, str_from_path


class TestValueEx3(ExIn, Tuple[InformedGraphSearchProblem, str]):
    def str_id(self) -> str:
        return str(self[1])


def ex3_evaluation(ex_in: TestValueEx3, ex_out=None) -> Report:
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
            wG._G, ax=ax, node_color=NodeColors.default, edge_color=EdgeColors.default, node_edgecolor="k", show=False
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
        for j, heuristic in enumerate(Heuristic):
            # Ground truth
            msg = f"Start: {query[0]},\tGoal: {query[1]}\n"
            search_algo = informed_graph_search_algo[algo_name]()
            rfig = r.figure(cols=2)
            heuristic = None if algo_name == 'UniformCostSearch' else heuristic
            
            # Your algo
            start = process_time()
            path = search_algo.path(wG, query[0], query[1], heuristic)
            solve_time = process_time() - start
            if path:
                path_str = str_from_path(path)
                msg += f"Your path: {path_str}\n"
                path_cost = compute_path_cost(wG, path)
                msg += f"Your path cost:\t{path_cost:.2f}\n"
                with rfig.plot(nid=f"YourPath{i}-{algo_name}-{heuristic}", mime=MIME_PDF, figsize=figsize) as _:
                    ax = plt.gca()
                    ox.plot_graph(wG._G, ax=ax, node_color=nc, node_edgecolor="k", edge_color=ec, show=False, close =False)
                    ox.plot_graph_route(wG._G, route=path, ax=ax, orig_dest_size=0, route_linewidth=1, show=False, close=False)
            else:
                msg += "Your algo did not find any path.\n"
                path_cost = None
                path = []

            # ground truths
            gt_path = ex_out[i][j]
            # compare to ground truth only for admissible heuristic
            if gt_path is not None and heuristic not in [Heuristic.MANHATTAN, Heuristic.INADMISSIBLE]:
                # Compute gt path cost
                gt_path_cost = compute_path_cost(wG, gt_path)
                gt_path_str = str_from_path(gt_path)
                msg += f"Ground truth path: {gt_path_str}\n"
                msg += f"Ground truth path cost: {gt_path_cost:.2f}\n"
                # Plot ground truth
                with rfig.plot(nid=f"GroundTruth{i}-{algo_name}-{heuristic}", mime=MIME_PDF, figsize=figsize) as _:
                    ax = plt.gca()
                    ox.plot_graph(wG._G, ax=ax, node_color=nc, node_edgecolor="k", edge_color=ec, show=False, close =False)
                    ox.plot_graph_route(wG._G, route=gt_path, ax=ax, orig_dest_size=0, route_linewidth=1, show=False, close=False)
                # Compare your algo to ground truth
                if  gt_path_cost == path_cost:
                    accuracy.append(1.)
                else:
                    accuracy.append(0.)
                solve_times.append(solve_time)
            else:
                msg += "Ground truth: Solution not given"
                
            heuristic_str = str(heuristic)[10:] if heuristic is not None else "None"
            r.text(f"{algo_name}-query{i}-{heuristic_str}", text=remove_escapes(msg))

            if algo_name == 'UniformCostSearch':
                break
    
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