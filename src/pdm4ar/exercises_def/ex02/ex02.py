from typing import Any, Sequence
from time import process_time
import os
import pathlib
import pickle
import shutil
from collections import OrderedDict

from reprep import Report, MIME_PNG, DataNode
from zuper_commons.text import remove_escapes
from matplotlib import pyplot as plt
from toolz import sliding_window

from pdm4ar.exercises.ex02 import graph_search_algo
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex02.data import *
from pdm4ar.exercises_def.structures import PerformanceResults
from pdm4ar.exercises_def.structures import out_dir


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

class GraphImageCache:
    """
    Generating images of graphs is extremely slow (over 90% of evaluation time).
    Moreover, many of the generated images are the same, within a single run
    and between multiple runs. This class manages images of graphs, so that identical
    graphs are not redrawn.

    The cache itself maps "graph encodings" to image ids. Then, there is a folder
    which contains images of the graphs, whose file names correspond to the image ids.

    We must include the node color and edge colors as attributes on the graphs.
    This is so that if two graphs have different node colors, they will have different
    encodings, and the GraphImageCache will know to redraw them.

    It is possible that the cache enters an inconsistent state if the user interrupts the
    program after adding/removing a file, but before the cache data has been updated.
    We try to prevent this from happening by scheduling these writes as close together
    as possible, and saving the cache state whenever we modify it. However, inconsistency
    is still possible. Therefore, every time we load the cache, we check if it matches the
    directory state, and if not, clear the cache and start over.
    """

    CACHE_SIZE = 100

    def __init__(self):
        cache_dir = pathlib.Path(out_dir("02")) / "cache"
        cache_file = cache_dir / "graph_data.pickle"
        create_new_cache = False

        # If there is cache data present, use it to fill in our field values.
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.__dict__ = pickle.load(f).__dict__

            if not self.consistency_check(cache_dir):
                shutil.rmtree(cache_dir)
                create_new_cache = True
        else:
            create_new_cache = True

        self.cache_dir = cache_dir
        self.cache_file = cache_file

        if create_new_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = OrderedDict()
            self.counter = 0

    def consistency_check(self, cache_dir) -> bool:
        """It is possible that the cache enters an inconsistent state if the program is interrupted
        between the time that someone writes/deletes and image and saves the cache data. We try to
        make this unlikely, by grouping these actions close together. However, if it does occur,
        we just delete the cache and start over :(
        """
        image_file_names = sorted(os.listdir(cache_dir))[:-1]  # everything except the pickle file
        if len(image_file_names) != len(self.cache):
            return False
        for image_id in self.cache.values():
            if f"{image_id}.png" not in image_file_names:
                return False
        return True

    def create_graph_image_node(self, graph: nx.Graph, name: str, pos, figsize) -> DataNode:
        """
        Create an image node, containing the image of the graph. When creating
        the html report, a parent node can add this image node as a child.
        """
        key = GraphImageCache.graph_encoding(graph)
        if key in self.cache:
            self.cache.move_to_end(key, last=True)
        else:
            # Make room in the cache, if necessary
            if len(self.cache) >= GraphImageCache.CACHE_SIZE:
                self.remove_oldest_graph_from_cache()

            self.add_graph_to_cache(graph, key, pos, figsize)

        # Create the image node.
        # In order to do this, we have to read in the graph image data
        image_id = self.cache[key]
        image_file = self.cache_dir / self.image_file(image_id)
        with open(image_file, "rb") as f:
            image_bytes = f.read()

        image_node = DataNode(nid=name, data=image_bytes, mime=MIME_PNG)
        return image_node

    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self, f)

    def remove_oldest_graph_from_cache(self):
        # pop the first item in the OrderedDict
        _, image_id = self.cache.popitem(last=False)

        # Delete the corresponding file
        oldest_graph_file = self.image_file(image_id)
        assert os.path.exists(oldest_graph_file)
        os.remove(oldest_graph_file)
        self.save()

    def add_graph_to_cache(self, graph: nx.Graph, graph_encoding: str, pos, figsize):
        # Draw the graph: recreate the node/edge color info based on the attributes
        # stored on the graph components
        node_colors = [graph.nodes[u]["node_color"] for u in graph.nodes]
        edge_colors = [graph.edges[u, v]["edge_color"] for (u, v) in graph.edges]
        nx.draw(graph, node_color=node_colors, edge_color=edge_colors, pos=pos, with_labels=True)
        plt.savefig(self.image_file(self.counter), pil_kwargs={"figsize":figsize})

        # add the graph data to our cache lookup
        self.cache[graph_encoding] = self.counter
        self.counter += 1
        self.save()

    def image_file(self, image_id):
        return self.cache_dir / f"{image_id}.png"

    @staticmethod
    def graph_encoding(graph: nx.Graph) -> str:
        # We need a way to convert graphs to "keys", that we can use as lookups
        # for the cache. These encodings should encorporate all the data that
        # uniquely defines a graph, and keys should only be equal if their
        # graphs are equal. It turns out, we can use python's pickle encoding
        # for this purpose
        return str(pickle.dumps(graph))

@dataclass(frozen=True)
class Ex02PerformanceResult(PerformanceResults):
    accuracy: float
    solve_time: float

    def __post__init__(self):
        assert 0 <= self.accuracy <= 1, self.accuracy
        assert self.solve_time >= 0, self.solve_time

def str_from_path(path:Path) -> str:
    return "".join(list(map(lambda u: f"{u}->", path)))[:-2]


def ex2_evaluation(ex_in, ex_out=None, plotGraph=True) -> Tuple[Ex02PerformanceResult, Report]:
    # draw graph
    graph_search_prob, algo_name = ex_in
    test_graph = graph_search_prob.graph
    test_queries = graph_search_prob.queries
    graph_id = graph_search_prob.graph_id

    # init rep with *unique* string id
    r = Report(f"Exercise2-{algo_name}-{graph_id}")
    cache = GraphImageCache()

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

    default_node_colors = {n: NodeColors.default for n in G}
    default_edge_colors = {(u, v): EdgeColors.default for (u, v) in G.edges()}
    nx.set_node_attributes(G, values=default_node_colors, name="node_color")
    nx.set_edge_attributes(G, values=default_edge_colors, name="edge_color")

    graph_image = cache.create_graph_image_node(G, "Graph", pos, figsize)
    rfig.add_child(graph_image)

    # run algo
    r.section(f"{algo_name}")
    accuracy = []
    solve_times = []
    for i, query in enumerate(test_queries):
        # Set all edge color attribute to black
        for e in G.edges():
            G[e[0]][e[1]]["color"] = EdgeColors.default
        
        rfig = r.figure(cols=2)

        # Your algo
        search_algo = graph_search_algo[algo_name]()
        start_time = process_time()
        path, opened = search_algo.search(test_graph, query[0], query[1])
        solve_time = process_time() - start_time
        # check path
        if path:
            path_str = str_from_path(path)
            path_edges = list(sliding_window(2, path))
        else:
            path_str = "No path"
            path_edges = []
        # check opened
        if opened:
            opened_str = str_from_path(opened)
        else:
            opened_str = "No opened node"

        # output message
        msg = f"Start: {query[0]},\tGoal: {query[1]}\n"

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
            if correct == 2:
                msg += "Student solution : CORRECT\n"
            else:
                msg += "Student solution : WRONG\n"
        else:
            gt_path_edges = []
            gt_path = []
            gt_path_str = "Solution not given"
            gt_opened_str = "Solution not given"

        msg += f"Your algo path: {path_str}\n"
        msg += f"Your algo opened nodes: {opened_str}\n"

        msg += f"Ground truth path: {gt_path_str} \n"
        msg += f"Ground truth opened nodes: {gt_opened_str}\n \n"

        r.text(f"{algo_name}-query{i}", text=remove_escapes(msg))

        # Plot graphs
        if plotGraph:
            node_colors = default_node_colors.copy()
            node_colors[query[0]] = NodeColors.start
            node_colors[query[1]] = NodeColors.goal
            edge_colors = {
                (u, v): EdgeColors.path if (u, v) in path_edges else EdgeColors.default \
                    for (u, v) in G.edges()
            }
            nx.set_node_attributes(G, values=node_colors, name="node_color")
            nx.set_edge_attributes(G, values=edge_colors, name="edge_color")
            image_node = cache.create_graph_image_node(G, f"Path{i}", pos, figsize)
            rfig.add_child(image_node)

            edge_colors = {
                (u, v): EdgeColors.path if (u, v) in gt_path_edges else EdgeColors.default \
                    for (u, v) in G.edges()
            }
            nx.set_edge_attributes(G, values=edge_colors, name="edge_color")
            image_node = cache.create_graph_image_node(G, f"GroundTruth{i}", pos, figsize)
            rfig.add_child(image_node)

    cache.save()

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
