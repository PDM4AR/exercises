from typing import Any, Sequence

from reprep import Report, MIME_PDF
from zuper_commons.text import remove_escapes
from matplotlib import pyplot as plt
from toolz import sliding_window
from time import process_time

from pdm4ar.exercises.ex02 import graph_search_algo
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex02.data import *
from pdm4ar.exercises_def.structures import PerformanceResults


@dataclass(frozen=True)
class NodeColors:
    default: str = "cyan"
    start: str = "orange"
    goal: str = "green"


@dataclass(frozen=True)
class EdgeColors:
    default: str = "dimgray"
    path: str = "red"
    gt_path: str = "green"


class TestValueEx2(ExIn, Tuple[GraphSearchProblem, str]):
    def str_id(self) -> str:
        return str(self[1])


@dataclass(frozen=True)
class Ex02PerformanceResult(PerformanceResults):
    accuracy: float
    solve_time: float

    def __post__init__(self):
        assert 0 <= self.accuracy <= 1, self.accuracy
        assert self.solve_time >= 0, self.solve_time

def str_from_path(path:Path) -> str:
    return "".join(list(map(lambda u: f"{u}->", path)))[:-2]


def ex2_evaluation(ex_in, ex_out=None) -> Tuple[Ex02PerformanceResult, Report]:
    # draw graph
    graph_search_prob, algo_name = ex_in
    test_graph = graph_search_prob.graph
    test_queries = graph_search_prob.queries
    graph_id = graph_search_prob.graph_id

    # init rep with *unique* string id
    r = Report(f"Exercise2-{algo_name}-{graph_id}")

    G = nx.DiGraph()
    G.add_nodes_from(test_graph.keys())
    pic_size = max(10, int(G.number_of_nodes() / 10))
    figsize = (pic_size, pic_size)
    for n, successors in test_graph.items():
        G.add_edges_from(
                product(
                        [
                            n,
                        ],
                        successors,
                )
        )
    # draw graph
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        pos = nx.spring_layout(G, seed=1)
    rfig = r.figure(cols=1)
    with rfig.plot(nid="Graph", mime=MIME_PDF, figsize=figsize) as _:
        nx.draw(G, pos=pos, with_labels=True, node_color=NodeColors.default)

    # run algo
    r.section(f"{algo_name}")
    accuracy = []
    solve_times = []
    for i, query in enumerate(test_queries):
        # Set all edge color attribute to black
        for e in G.edges():
            G[e[0]][e[1]]["color"] = EdgeColors.default
        msg = f"Start: {query[0]},\tGoal: {query[1]}\n"
        rfig = r.figure(cols=2)

        # Your algo
        search_algo = graph_search_algo[algo_name]()
        start_time = process_time()
        path, opened = search_algo.search(test_graph, query[0], query[1])
        solve_time = process_time() - start_time
        if path and opened:
            path_str = str_from_path(path)
            opened_str = str_from_path(opened)
            path_edges = list(sliding_window(2, path))
        else:
            path_str = "No path"
            try: 
                opened_str = str_from_path(opened)
            except:
                opened_str = "No opened node"
            path_edges = []

        msg += f"Your algo path: {path_str}\n"
        msg += f"Your algo opened nodes: {opened_str}\n"

        # Ground truth
        expected_result = ex_out[i]
        if expected_result is not None:
            gt_path, gt_opened = expected_result
            correct = (path == gt_path) + (opened == gt_opened)
            accuracy.append(correct / 2)
            solve_times.append(solve_time)
            gt_path_str = str_from_path(gt_path) if len(gt_path) > 0 else "No path"
            gt_opened_str = str_from_path(gt_opened)
            gt_path_edges = list(sliding_window(2, gt_path))
        else:
            gt_path_edges = []
            gt_path = []
            gt_path_str = "Solution not given"
            gt_opened_str = "Solution not given"

        msg += f"Ground truth path: {gt_path_str} \n"
        msg += f"Ground truth opened nodes: {gt_opened_str}\n \n"

        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))

        # Plot graphs
        with rfig.plot(nid=f"Path{i}", mime=MIME_PDF, figsize=figsize) as _:
            node_colors = [
                NodeColors.start if n == query[0] else (NodeColors.goal if n == query[1] else NodeColors.default)
                for n in G
            ]
            ax = plt.gca()
            nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_colors)
            nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=EdgeColors.default)
            nx.draw_networkx_edges(G, ax=ax, pos=pos, edgelist=path_edges, edge_color=EdgeColors.path)
            nx.draw_networkx_labels(G, ax=ax, pos=pos)

        with rfig.plot(nid=f"GroundTruth{i}", mime=MIME_PDF, figsize=figsize) as _:
            node_colors = [
                NodeColors.start if n == query[0] else (NodeColors.goal if n == query[1] else NodeColors.default)
                for n in G
            ]
            ax = plt.gca()
            nx.draw_networkx_nodes(G, ax=ax, pos=pos, node_color=node_colors)
            nx.draw_networkx_edges(G, ax=ax, pos=pos, edge_color=EdgeColors.default)
            nx.draw_networkx_edges(G, ax=ax, pos=pos, edgelist=gt_path_edges, edge_color=EdgeColors.path)
            nx.draw_networkx_labels(G, ax=ax, pos=pos)

    # aggregate performance of each query
    query_perf = list(map(Ex02PerformanceResult, accuracy, solve_times))
    perf = ex2_perf_aggregator(query_perf)
    return perf, r


def ex2_perf_aggregator(perf: Sequence[Ex02PerformanceResult]) -> Ex02PerformanceResult:
    # perfomance for valid results
    valid_acc = [p.accuracy for p in perf]
    valid_time = [p.solve_time for p in perf]
    try:
        # average accuracy
        avg_acc = sum(valid_acc) / float(len(valid_acc))
        # average solve time
        avg_time = sum(valid_time) / float(len(valid_time))
    except ZeroDivisionError:
        # None if gt wasn't provided
        avg_acc = 0
        avg_time = 0

    return Ex02PerformanceResult(accuracy=avg_acc, solve_time=avg_time)


def get_exercise2() -> Exercise:
    graph_search_problems = get_graph_search_problems(n_seed=4)
    expected_results = ex2_get_expected_results() 
    graph_search_algos = graph_search_algo.keys()

    test_values = list()
    for ab in product(graph_search_problems, graph_search_algos):
        test_values.append(TestValueEx2(ab))

    return Exercise[TestValueEx2, Any](
            desc='This exercise is about graph search',
            evaluation_fun=ex2_evaluation,
            perf_aggregator=ex2_perf_aggregator,
            test_values=test_values,
            expected_results=expected_results,
    )
