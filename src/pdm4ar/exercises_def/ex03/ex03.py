from itertools import product
from typing import Tuple, Any, Sequence
from dataclasses import dataclass

import osmnx as ox
from time import process_time
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF, Node
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def import Exercise, NodeColors, EdgeColors, ExIn
from pdm4ar.exercises_def.structures import PerformanceResults
from pdm4ar.exercises.ex03 import informed_graph_search_algo, compute_path_cost, \
    UniformCostSearch, Astar
from pdm4ar.exercises_def.ex02 import str_from_path
from pdm4ar.exercises.ex02.structures import X
from pdm4ar.exercises_def.ex03.data import ex3_get_expected_results, get_test_informed_gsproblem, \
    InformedGraphSearchProblem


class TestValueEx3(ExIn, Tuple[InformedGraphSearchProblem, str]):
    def str_id(self) -> str:
        return str(self[1])

@dataclass(frozen=True)
class Ex03PerformanceResult(PerformanceResults):
    accuracy: float
    solve_time: float
    heuristic_efficiency: float = 0

    def __post__init__(self):
        assert 0 <= self.accuracy <= 1, self.accuracy
        assert self.solve_time >= 0, self.solve_time
        assert 0 <= self.heuristic_efficiency, self.heuristic_efficiency

def ex3_evaluation(ex_in: TestValueEx3, ex_out=None, plotGraph=True) -> Tuple[Ex03PerformanceResult, Report]:
    # ex properties
    prob, (algo_name, heuristic_count_fn) = ex_in
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
    heuristic_performance = []
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
        heuristic_count = heuristic_count_fn(search_algo, query[0], query[1])

        if path:
            path_str = str_from_path(path)
            path_cost = compute_path_cost(wG, path)
            if plotGraph:
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
        gt_path, trivial_heuristic_count = ex_out[i]
        # compare to ground truth only for admissible heuristic
        if gt_path is not None:
            # Compute gt path cost
            gt_path_cost = compute_path_cost(wG, gt_path)
            gt_path_str = str_from_path(gt_path)
            # Plot ground truth
            if plotGraph:
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

            if algo_name == Astar.__name__:
                # This section of code can be reached under two circumstances.
                # 1. The server is evaluating the code. Then the trivial_heuristic_count
                #    is a strictly positive number, which is calculated using the server's
                #    implementation of Astar with a heuristic of 0
                # 2. The code is being evaluated locally. This is indicated by setting
                #    the trivial_heuristic_count to 0. To find the true value of the
                #    trivial_heuristic_count, we must rerun the student's Astar implementation
                #    with a heuristic of 0.

                if trivial_heuristic_count == 0:
                    # We must calculate the trivial heuristic count.
                    # Tell the student's algorithm to use the trivial heuristic rather than the one
                    # the implemented.
                    search_algo.heuristic_counter = 0
                    search_algo.use_trivial_heuristic = True
                    # Rerun Astar, counting how many times the heuristic was invoked
                    search_algo.path(query[0], query[1])
                    trivial_heuristic_count = heuristic_count_fn(search_algo, query[0], query[1])

                if trivial_heuristic_count == 0:
                    # This case is only hit of the student never calls the heuristic.
                    heuristic_performance.append(float('inf'))
                else:
                    heuristic_performance.append(heuristic_count / trivial_heuristic_count)
            else:
                heuristic_performance.append(0.)
        else:
            gt_path_str = "Solution not given"

        # output path to report
        msg += f"Ground truth path: {gt_path_str}\n"
        msg += f"Ground truth path cost: {gt_path_cost:.2f}\n"

        msg += f"Your path: {path_str}\n"
        msg += f"Your path cost:\t{path_cost:.2f}\n"

        if algo_name == Astar.__name__:
            msg += f"Your heuristic call counter: {heuristic_count}\n"
            msg += f"Trivial heuristic call counter: {trivial_heuristic_count}\n"

        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))

    # aggregate performance of each query
    query_perf = list(map(Ex03PerformanceResult, accuracy, solve_times, heuristic_performance))
    perf = ex3_perf_aggregator(query_perf)
    return perf, r

def ex3_perf_aggregator(perf: Sequence[Ex03PerformanceResult]) -> Ex03PerformanceResult:
    # Very similar to ex2 perf aggregator, except now we include the heuristic performance

    # perfomance for valid results
    valid_acc = [p.accuracy for p in perf]
    valid_time = [p.solve_time for p in perf]
    valid_heuristic_efficiency = [p.heuristic_efficiency for p in perf if p.heuristic_efficiency != 0]

    avgs = []
    for valid_result in [valid_acc, valid_time, valid_heuristic_efficiency]:
        try:
            avg_result = sum(valid_result) / float(len(valid_result))
        except ZeroDivisionError:
            avg_result = 0.
        avgs.append(avg_result)

    return Ex03PerformanceResult(accuracy=avgs[0], solve_time=avgs[1], heuristic_efficiency=avgs[2])

def get_exercise3() -> Exercise:
    test_wgraphs = get_test_informed_gsproblem(n_queries=1, n_seed=4)
    expected_results = ex3_get_expected_results()
    test_values = list()

    def uniform_cost_heuristic_counter(search_algo: UniformCostSearch, start: X, goal: X) -> int:
        # There is no heuristic in UCS, so we just return 0
        return 0

    def astar_heuristic_counter(search_algo: Astar, start: X, goal: X) -> int:
        return search_algo.heuristic_counter

    algos = [(UniformCostSearch.__name__, uniform_cost_heuristic_counter),
             (Astar.__name__, astar_heuristic_counter)]

    for ab in product(test_wgraphs, algos):
        test_values.append(TestValueEx3(ab))

    return Exercise[TestValueEx3, Any](
            desc='This exercise is about graph search',
            evaluation_fun=ex3_evaluation,
            perf_aggregator=ex3_perf_aggregator,
            test_values=test_values,
            expected_results=expected_results,
    )
