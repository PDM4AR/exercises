from itertools import product
from typing import Tuple, Any, Sequence, Callable
from dataclasses import dataclass

import osmnx as ox
from time import process_time
from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF, Node
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def import Exercise, NodeColors, EdgeColors
from pdm4ar.exercises_def.structures import PerformanceResults
from pdm4ar.exercises.ex03 import (
    informed_graph_search_algo,
    compute_path_cost,
    UniformCostSearch,
    Astar,
)
from pdm4ar.exercises_def.ex02 import str_from_path
from pdm4ar.exercises.ex02.structures import X
from pdm4ar.exercises_def.ex03.data import (
    ex3_compute_expected_results,
    get_test_informed_gsproblem,
    graph_dimensions,
    find_center_of_cities,
    TestValueEx3
)


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
    prob = ex_in.problem
    algo_name = ex_in.algo_name
    heuristic_count_fn = ex_in.h_count_fn
    wG = prob.graph
    test_queries = prob.queries
    ec = [EdgeColors.default for uv in wG._G.edges]
    # init report
    r = Report(f"Exercise3-{algo_name}-{prob.graph_id}")
    # draw graph
    figsize = None
    rfig = r.figure(cols=2)
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=figsize) as _:
        ax = plt.gca()
        ox.plot_graph(
            wG._G,
            ax=ax,
            node_color=NodeColors.default,
            edge_color=EdgeColors.default,
            node_edgecolor="k",
            show=False,
        )

    # initialisation of performance
    r.section(f"{algo_name}")
    solve_times = []
    accuracy = []
    heuristic_performance = []
    search_algo = informed_graph_search_algo[algo_name](wG)
    # Validate implementation
    validation_wrapper = ex_in.impl_validate_func_wrapper
    disallowed_deps = ex_in.disallowed_dependencies
    if validation_wrapper is not None and disallowed_deps is not None:
        data = next(iter(test_queries))                                     # Use one of the available queries for validation
        check = validation_wrapper(search_algo.path, disallowed_deps)
        called_funcs = check(*data)
        if called_funcs:
            validation_details = []
            validation_details.append("Implementation validation failed. Disallowed dependencies detected:")
            for record in called_funcs:
                validation_details.append(
                    f"  - Library: {record['library']}, "
                    f"Function: {record['func_name']}, Line: {record['lineno']}, File: {record['filename']}"
                )
            r.text(f"{algo_name}-validation-query{data}", "\n".join(validation_details))
            return Ex03PerformanceResult(accuracy=0.0, solve_time=0.0, heuristic_efficiency=float("inf")), r
    # run algo looping over all queries
    for i, query in enumerate(test_queries):
        nc = [
            (NodeColors.start if n == query[0] else (NodeColors.goal if n == query[1] else NodeColors.default))
            for n in wG._G
        ]
        # Ground truth
        msg = f"Start: {query[0]},\tGoal: {query[1]}\n"
        search_algo = informed_graph_search_algo[algo_name](wG)             # new instance for each query
        rfig = r.figure(cols=2)
        # Your algo
        start = process_time()
        path = search_algo.path(query[0], query[1])
        solve_time = process_time() - start
        heuristic_count, path_heuristic_val = heuristic_count_fn(search_algo, query[0], query[1])
        # ground truths
        gt_path, trivial_heuristic_count = ex_out[i]
        if path:
            path_str = str_from_path(path)
            path_cost = compute_path_cost(wG, path)
            if plotGraph:
                # case 2 cities connected
                if graph_dimensions(wG._G)[0] > 1:
                    # print("printing double")
                    centers = find_center_of_cities(wG._G)
                    print(f"center city {centers[0]}")
                    with rfig.plot(nid=f"YourPath{i}-{algo_name}", mime=MIME_PDF, figsize=figsize) as _:
                        ax = plt.gca()
                        # function needed to display one of the combined
                        # city around its center
                        bbox = ox.utils_geo.bbox_from_point(centers[0], 1500, project_utm=False, return_crs=False)
                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            bbox=bbox,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        if len(path) > 0:
                            ox.plot_graph_route(
                                wG._G,
                                route=path,
                                ax=ax,
                                orig_dest_size=0,
                                route_linewidth=5,
                                show=False,
                                close=False,
                            )

                    with rfig.plot(
                        nid=f"YourPath{i}-{algo_name}-{2}",
                        mime=MIME_PDF,
                        figsize=figsize,
                    ) as _:
                        ax = plt.gca()
                        bbox = ox.utils_geo.bbox_from_point(
                            centers[1],
                            1500,
                            project_utm=False,
                            return_crs=False,
                        )
                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            bbox=bbox,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        if len(path) > 0:
                            ox.plot_graph_route(
                                wG._G,
                                route=path,
                                ax=ax,
                                orig_dest_size=0,
                                route_linewidth=5,
                                show=False,
                                close=False,
                            )
                else:
                    # standard case
                    with rfig.plot(nid=f"YourPath{i}-{algo_name}", mime=MIME_PDF, figsize=figsize) as _:
                        ax = plt.gca()
                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        ox.plot_graph_route(
                            wG._G,
                            route=path,
                            ax=ax,
                            orig_dest_size=0,
                            route_linewidth=5,
                            show=False,
                            close=False,
                        )
        elif not gt_path:                  # No ground truth path found, so it doesn't exist --> set default values
            path_str = "Your algo did not find any path, because it does not exist."
            gt_path_str = "Solution not given"
            path_cost = 0.0
            gt_path_cost = 0.0
        else:
            path_str = "Your algo did not find any path."
            path_cost = float("inf")
            path = []
        # compare to ground truth only for admissible heuristic
        if gt_path:                 # if gt_path is not empty
            # Compute gt path cost
            gt_path_cost = compute_path_cost(wG, gt_path)
            gt_path_str = str_from_path(gt_path)
            # Plot ground truth
            if plotGraph:
                if graph_dimensions(wG._G)[0] > 1:
                    # print("printing double city")
                    centers = find_center_of_cities(wG._G)
                    with rfig.plot(
                        nid=f"GroundTruth{i}-{algo_name}",
                        mime=MIME_PDF,
                        figsize=figsize,
                    ) as _:
                        ax = plt.gca()
                        bbox = ox.utils_geo.bbox_from_point(centers[0], 1500, project_utm=False, return_crs=False)
                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            bbox=bbox,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        if len(gt_path) > 0:
                            ox.plot_graph_route(
                                wG._G,
                                route=gt_path,
                                ax=ax,
                                orig_dest_size=0,
                                route_linewidth=5,
                                show=False,
                                close=False,
                            )

                    with rfig.plot(
                        nid=f"GroundTruth{i}-{algo_name}-{2}",
                        mime=MIME_PDF,
                        figsize=figsize,
                    ) as _:
                        ax = plt.gca()
                        bbox = ox.utils_geo.bbox_from_point(
                            centers[1],
                            1500,
                            project_utm=False,
                            return_crs=False,
                        )

                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            bbox=bbox,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        if len(gt_path) > 0:
                            ox.plot_graph_route(
                                wG._G,
                                route=gt_path,
                                ax=ax,
                                orig_dest_size=0,
                                route_linewidth=5,
                                show=False,
                                close=False,
                            )
                else:
                    with rfig.plot(
                        nid=f"GroundTruth{i}-{algo_name}",
                        mime=MIME_PDF,
                        figsize=figsize,
                    ) as _:
                        ax = plt.gca()
                        ox.plot_graph(
                            wG._G,
                            ax=ax,
                            node_color=nc,
                            node_edgecolor="k",
                            edge_color=ec,
                            show=False,
                            close=False,
                        )
                        ox.plot_graph_route(
                            wG._G,
                            route=gt_path,
                            ax=ax,
                            orig_dest_size=0,
                            route_linewidth=5,
                            show=False,
                            close=False,
                        )
            # Compare your algo to ground truth
            if gt_path_cost == path_cost:
                # Validate student's heuristic
                if path_heuristic_val is not None:
                    if path_heuristic_val:  # non-empty
                        cross_path_cost = compute_path_cost(wG, path_heuristic_val)
                    else:  # empty path
                        cross_path_cost = 0.0

                    if cross_path_cost != gt_path_cost:
                        msg += (
                            "CORRECT solution but heuristic function is NOT ADMISSIBLE outside local astar: "
                            f"astar + local heuristic cost {cross_path_cost} != astar + admissible heuristic cost {gt_path_cost}\n"
                        )
                        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))
                        return Ex03PerformanceResult(
                            accuracy=0.0,
                            solve_time=solve_time,
                            heuristic_efficiency=float("inf"),
                        ), r

                # validated
                accuracy.append(1.0)
                msg += "Student solution : CORRECT\n"
            else:
                accuracy.append(0.0)
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
                    trivial_heuristic_count, _ = heuristic_count_fn(search_algo, query[0], query[1])
                if len(gt_path) == 1:
                    # Case when start = goal
                    heuristic_performance.append(0.0)
                elif trivial_heuristic_count == 0:
                    # This case is only hit of the student never calls the heuristic.
                    heuristic_performance.append(float("inf"))
                else:
                    heuristic_performance.append(heuristic_count / trivial_heuristic_count)
            else:
                heuristic_performance.append(0.0)

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
            avg_result = 0.0
        avgs.append(avg_result)

    return Ex03PerformanceResult(accuracy=avgs[0], solve_time=avgs[1], heuristic_efficiency=avgs[2])


def validate_impl_wrapper(func: Callable, disallowed_dependencies: dict[str, set[str]]) -> Callable:
    called_funcs = []
    detected_funcs = set()  # Track already detected libraries

    def trace_calls(frame, event, arg):  # pylint: disable=unused-argument
        import traceback  # pylint: disable=import-outside-toplevel

        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        func_name = frame.f_code.co_name
        for lib in disallowed_dependencies:
            if module.startswith(lib) and (func_name in disallowed_dependencies[lib] or not disallowed_dependencies[lib]):
                identifier = (lib, func_name)
                if identifier not in detected_funcs:
                    # Only record each function once
                    detected_funcs.add(identifier)

                    traces = traceback.extract_stack(frame)
                    for trace in reversed(traces):
                        if trace.name == func.__name__:
                            called_funcs.append(
                                {
                                    "library": lib,
                                    "func_name": trace.name,
                                    "lineno": trace.lineno,
                                    "filename": trace.filename.split("/")[-1],  # Get the filename only
                                }
                            )
                            break
        return trace_calls

    def wrapper(*args, **kwargs):
        import sys  # pylint: disable=import-outside-toplevel

        sys.setprofile(trace_calls)
        try:
            func(*args, **kwargs)
        finally:
            sys.setprofile(None)
        return called_funcs

    return wrapper


def get_exercise3() -> Exercise:
    disallowed_dependencies = {"networkx": {"astar_path", "shortest_path"}, 
                               "ctypes": set()}    # ctypes is disallowed in its entirety

    test_wgraphs = get_test_informed_gsproblem(n_queries=1, n_seed=4)
    test_values = list()

    def uniform_cost_heuristic_counter(search_algo: UniformCostSearch, start: X, goal: X) -> Tuple[int, list]:
        # There is no heuristic in UCS, so we just return 0
        return 0, None

    def astar_heuristic_counter(search_algo: Astar, start: X, goal: X) -> Tuple[int, list]:
        return search_algo.heuristic_counter, None

    algos = [
        (UniformCostSearch.__name__, uniform_cost_heuristic_counter),
        (Astar.__name__, astar_heuristic_counter),
    ]

    for prob, (algo_name, algo_func) in product(test_wgraphs, algos):
        test_values.append(TestValueEx3(
            problem=prob,
            algo_name=algo_name,
            h_count_fn=algo_func,
            impl_validate_func_wrapper=validate_impl_wrapper,     # else None
            disallowed_dependencies=disallowed_dependencies       # else None
    ))


    expected_results = ex3_compute_expected_results(test_values)

    return Exercise[TestValueEx3, Any](
        desc="This exercise is about graph search",
        evaluation_fun=ex3_evaluation,
        perf_aggregator=ex3_perf_aggregator,
        test_values=test_values,
        expected_results=expected_results,
    )
