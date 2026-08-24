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
    UniformCostSearch,
    BidirectionalUniformCostSearch,
    Astar,
)
from pdm4ar.exercises_def.ex02 import str_from_path
from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph
from pdm4ar.exercises_def.ex03.instrumentation import instrument_weight_lookups
from pdm4ar.exercises_def.ex03.data import (
    ex3_compute_expected_results,
    get_test_informed_gsproblem,
    graph_dimensions,
    find_center_of_cities,
    TestValueEx3,
)


@dataclass(frozen=True)
class Ex03PerformanceResult(PerformanceResults):
    accuracy: float
    solve_time: float
    search_efficiency: float = 0
    weight_calls: int = 0
    reference_weight_calls: int = 0

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1, self.accuracy
        assert self.solve_time >= 0, self.solve_time
        assert 0 <= self.search_efficiency, self.search_efficiency
        assert self.weight_calls >= 0, self.weight_calls
        assert self.reference_weight_calls >= 0, self.reference_weight_calls


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total

def ex3_evaluation(ex_in: TestValueEx3, ex_out=None, plotGraph=True) -> Tuple[Ex03PerformanceResult, Report]:
    # ex properties
    prob = ex_in.problem
    algo_name = ex_in.algo_name
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
    weight_calls = []
    reference_weight_calls = []
    validation_graph, _ = instrument_weight_lookups(wG)
    search_algo = informed_graph_search_algo[algo_name](validation_graph)
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
            return Ex03PerformanceResult(accuracy=0.0, solve_time=0.0, search_efficiency=float("inf")), r
    # run algo looping over all queries
    for i, query in enumerate(test_queries):
        nc = [
            (NodeColors.start if n == query[0] else (NodeColors.goal if n == query[1] else NodeColors.default))
            for n in wG._G
        ]
        # Ground truth
        msg = f"Start: {query[0]},\tGoal: {query[1]}\n"
        instrumented_graph, counting_weights = instrument_weight_lookups(wG)
        search_algo = informed_graph_search_algo[algo_name](instrumented_graph)
        rfig = r.figure(cols=2)
        # Your algo
        start = process_time()
        path = search_algo.path(query[0], query[1])
        solve_time = process_time() - start
        student_weight_calls = counting_weights.count
        # ground truths
        expected = ex_out[i]
        gt_path = expected.path
        reference_ucs_weight_calls = expected.reference_ucs_weight_calls
        gt_path_str = str_from_path(gt_path) if gt_path else "Solution not given"
        gt_path_cost = compute_path_cost(wG, gt_path) if gt_path else 0.0
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

        is_correct = gt_path == path
        accuracy.append(1.0 if is_correct else 0.0)
        solve_times.append(solve_time)
        msg += "Student solution : CORRECT\n" if is_correct else "Student solution : WRONG\n"

        # Efficiency is aggregated only for correct, non-trivial Bi-UCS and A* queries.
        include_in_efficiency = algo_name in {
            BidirectionalUniformCostSearch.__name__,
            Astar.__name__,
        }
        if include_in_efficiency and is_correct and reference_ucs_weight_calls > 0:
            weight_calls.append(student_weight_calls)
            reference_weight_calls.append(reference_ucs_weight_calls)
        else:
            weight_calls.append(0)
            reference_weight_calls.append(0)

        # output path to report
        msg += f"Ground truth path: {gt_path_str}\n"
        msg += f"Ground truth path cost: {gt_path_cost:.2f}\n"

        msg += f"Your path: {path_str}\n"
        msg += f"Your path cost:\t{path_cost:.2f}\n"

        msg += f"Your edge-weight accesses: {student_weight_calls}\n"
        msg += f"Reference UCS edge-weight accesses: {reference_ucs_weight_calls}\n"
        if is_correct and reference_ucs_weight_calls > 0:
            query_efficiency = student_weight_calls / reference_ucs_weight_calls
            msg += f"Search-efficiency ratio: {query_efficiency:.4f}\n"
            if not include_in_efficiency:
                msg += "Search-efficiency ratio: diagnostic only; UCS is excluded from aggregation\n"
        else:
            msg += "Search-efficiency ratio: not included in aggregation\n"

        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))

    # aggregate performance of each query
    query_perf = [
        Ex03PerformanceResult(
            accuracy=query_accuracy,
            solve_time=query_time,
            search_efficiency=(calls / reference_calls if reference_calls else 0.0),
            weight_calls=calls,
            reference_weight_calls=reference_calls,
        )
        for query_accuracy, query_time, calls, reference_calls in zip(
            accuracy, solve_times, weight_calls, reference_weight_calls
        )
    ]
    perf = ex3_perf_aggregator(query_perf)
    return perf, r


def ex3_perf_aggregator(perf: Sequence[Ex03PerformanceResult]) -> Ex03PerformanceResult:
    if not perf:
        return Ex03PerformanceResult(accuracy=0.0, solve_time=0.0)

    accuracy = sum(p.accuracy for p in perf) / len(perf)
    solve_time = sum(p.solve_time for p in perf) / len(perf)
    total_weight_calls = sum(p.weight_calls for p in perf)
    total_reference_calls = sum(p.reference_weight_calls for p in perf)
    search_efficiency = (
        total_weight_calls / total_reference_calls if total_reference_calls else 0.0
    )

    return Ex03PerformanceResult(
        accuracy=accuracy,
        solve_time=solve_time,
        search_efficiency=search_efficiency,
        weight_calls=total_weight_calls,
        reference_weight_calls=total_reference_calls,
    )


def _static_impl_violations(func: Callable) -> list[dict[str, Any]]:
    """Find source-level attempts to bypass the public weighted-graph API."""

    import ast  # pylint: disable=import-outside-toplevel
    import inspect  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    forbidden_attributes = {
        "_G",
        "weights",
        "_weights",
        "__dict__",
        "__code__",
        "__closure__",
        "__self__",
        "__globals__",
        "__getattribute__",
        "__subclasses__",
        "__traceback__",
        "_getframe",
        "ag_frame",
        "cr_frame",
        "f_back",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
    }
    forbidden_calls = {"eval", "exec", "globals", "locals", "vars", "__import__"}
    forbidden_imports = {"cloudpickle", "ctypes", "dill", "gc", "importlib", "inspect", "pickle", "sys"}
    forbidden_import_prefixes = ("pdm4ar.exercises_def", "pdm4ar_sol")

    source_path_str = inspect.getsourcefile(func)
    if source_path_str is None:
        return [
            {
                "library": "implementation validation",
                "func_name": "source unavailable",
                "lineno": 0,
                "filename": "<unknown>",
            }
        ]

    source_path = Path(source_path_str)
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        return [
            {
                "library": "implementation validation",
                "func_name": f"source inspection failed: {type(exc).__name__}",
                "lineno": 0,
                "filename": source_path.name,
            }
        ]

    violations: list[dict[str, Any]] = []
    detected: set[tuple[str, str, int]] = set()

    def add_violation(category: str, detail: str, lineno: int) -> None:
        identifier = (category, detail, lineno)
        if identifier in detected:
            return
        detected.add(identifier)
        violations.append(
            {
                "library": category,
                "func_name": detail,
                "lineno": lineno,
                "filename": source_path.name,
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
            add_violation("private graph/evaluator state", f"attribute .{node.attr}", node.lineno)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                add_violation("runtime introspection", f"call {node.func.id}()", node.lineno)
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in forbidden_attributes
            ):
                add_violation(
                    "private graph/evaluator state",
                    f"getattr(..., {node.args[1].value!r})",
                    node.lineno,
                )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module.split(".")[0] in forbidden_imports or module.startswith(forbidden_import_prefixes):
                    add_violation("disallowed import", module, node.lineno)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in forbidden_imports or module.startswith(forbidden_import_prefixes):
                add_violation("disallowed import", module, node.lineno)

    return violations


def validate_impl_wrapper(func: Callable, disallowed_dependencies: dict[str, set[str]]) -> Callable:
    called_funcs = []
    detected_funcs = set()  # Track already detected libraries
    static_violations = _static_impl_violations(func)
    effective_disallowed_dependencies = {
        **disallowed_dependencies,
        "ctypes": set(),
        "gc": set(),
        "importlib": set(),
        "inspect": set(),
        "sys": {"_getframe", "setprofile", "settrace"},
    }

    def trace_calls(frame, event, arg):  # pylint: disable=unused-argument
        import traceback  # pylint: disable=import-outside-toplevel

        if event not in ("call", "c_call"):
            return
        module = frame.f_globals.get("__name__", "")
        called_module = (getattr(arg, "__module__", "") or "") if event == "c_call" else ""
        func_name = getattr(arg, "__name__", frame.f_code.co_name) if event == "c_call" else frame.f_code.co_name
        for lib in effective_disallowed_dependencies:
            if module.startswith(lib) or called_module.startswith(lib):
                # print(f"Found call to {lib}")
                if (
                    not effective_disallowed_dependencies[lib]
                    or func_name in effective_disallowed_dependencies[lib]
                ):
                    # print(f"Detected disallowed dependency: {lib}.{func_name}")
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

        if static_violations:
            return static_violations
        sys.setprofile(trace_calls)
        try:
            func(*args, **kwargs)
        finally:
            sys.setprofile(None)
        return called_funcs

    return wrapper


def get_exercise3() -> Exercise:
    disallowed_dependencies = {"networkx": {"astar_path", "shortest_path", "dijkstra_path", "bidirectional_dijkstra"},
                               "ctypes": set()}    # ctypes is disallowed in its entirety

    test_wgraphs = get_test_informed_gsproblem(n_queries=1, n_seed=4)
    test_values = list()

    algos = [
        UniformCostSearch.__name__,
        BidirectionalUniformCostSearch.__name__,
        Astar.__name__,
    ]

    for prob, algo_name in product(test_wgraphs, algos):
        test_values.append(TestValueEx3(
            problem=prob,
            algo_name=algo_name,
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
