from dataclasses import dataclass
import pathlib
import pickle
from threading import local
from typing import Callable, Optional, Set

import osmnx as ox
from requests import get
from sklearn.cluster import KMeans
import pandas as pd
from frozendict import frozendict
from networkx import MultiDiGraph, compose, astar_path, NetworkXNoPath
import random
from collections import defaultdict

from pdm4ar.exercises.ex02.structures import Query, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed
from pdm4ar.exercises.ex03.local_queries import get_local_queries
from pdm4ar.exercises_def import networkx_2_adjacencylist, queries_from_adjacency, ExIn

_fast = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "escape",
    "track",
)
_slow = ("tertiary", "residential", "tertiary_link", "living_street")
_other = ("unclassified", "road", "service")


@dataclass
class InformedGraphSearchProblem:
    graph: WeightedGraph
    queries: set[Query]
    graph_id: str


@dataclass
class TestValueEx3(ExIn):
    problem: InformedGraphSearchProblem
    algo_name: str
    h_count_fn: Callable
    impl_validate_func_wrapper: Optional[Callable] = None
    disallowed_dependencies: Optional[Set[str]] = None
    def str_id(self) -> str:
        return str(self.algo_name)


def _find_speed(row) -> float:
    if row["highway"] in _fast:
        return TravelSpeed.HIGHWAY.value
    elif row["highway"] in _slow:
        return TravelSpeed.CITY.value
    elif row["highway"] in _other:
        return TravelSpeed.SECONDARY.value
    else:
        return TravelSpeed.PEDESTRIAN.value


def add_travel_time_weight(G: MultiDiGraph) -> MultiDiGraph:
    nodes, edges = ox.graph_to_gdfs(G)
    edges = edges.assign(speed=edges.apply(_find_speed, axis=1))
    edges["travel_time"] = edges["length"] / edges["speed"]
    UG = ox.graph_from_gdfs(nodes, edges)
    return UG


def networkx_2_weighted_graph(G: MultiDiGraph) -> WeightedGraph:
    G = add_travel_time_weight(G)
    adj = networkx_2_adjacencylist(G)
    weights = dict()
    for source, successors in adj.items():
        for dest in successors:
            min_weight = min([edge["travel_time"] for edge in G[source][dest].values()])
            assert isinstance(min_weight, float)
            assert min_weight > 0
            weights[(source, dest)] = min_weight
    wG = WeightedGraph(adj_list=adj, weights=frozendict(weights), _G=G)
    return wG


def download_gsproblems(test_maps=None, data_dir=None):
    # This is the function we used to download the city graphs for user tests.
    # Feel free to modify the code and download more graphs

    # Our tests were created using historical data from October 2022
    ox.settings.overpass_settings = (
        '[out:json][timeout:90][date:"2022-10-25T00:00:00Z"]'
    )

    if not test_maps:
        test_maps = [
            ("ny", "350 5th Ave, New York, New York", {"network_type": "drive"}),
            (
                "eth",
                "Rämistrasse 101, 8092 Zürich, Switzerland",
                {"network_type": "drive"},
            ),
            (
                "milan",
                "P.za del Duomo, 20122 Milano MI, Italy",
                {"network_type": "drive"},
            ),
        ]

    if not data_dir:
        data_dir = pathlib.Path(__file__).parent

    for graph_id, address, kwargs in test_maps:
        G_map = ox.graph_from_address(address, **kwargs)
        with open(data_dir / f"{graph_id}.pickle", "wb") as f:
            pickle.dump(G_map, f)


def get_test_informed_gsproblem(
    n_queries=1,
    n_seed=None,
    extra_test_graph_problems: list[InformedGraphSearchProblem] = [],
) -> list[InformedGraphSearchProblem]:
    data_dir = pathlib.Path(__file__).parent

    graph_ids = ["ny", "eth", "milan"]
    test_graphs = []
    for graph_id in graph_ids:
        with open(data_dir / f"{graph_id}.pickle", "rb") as f:
            test_graphs.append(pickle.load(f))

    # add travel time as weight
    test_graphs = map(add_travel_time_weight, test_graphs)
    test_wgraphs = map(networkx_2_weighted_graph, test_graphs)
    # convert graph to InformedGraphSearchProblem
    data_in: list[InformedGraphSearchProblem] = []
    for i, G in enumerate(test_wgraphs):
        id_graph = graph_ids[i]
        default_queries = queries_from_adjacency(G.adj_list, n=n_queries, n_seed=n_seed)  # set of queries
        local_queries = get_local_queries(G, id_graph)
        q = default_queries | local_queries
        p = InformedGraphSearchProblem(
            graph=G,
            queries=q,
            graph_id=id_graph,
        )
        data_in.append(p)
    # add InformedGraphSearchProblem for evaluation
    for extra_problem in extra_test_graph_problems:
        data_in.append(extra_problem)

    return data_in


def graph_dimensions(G: WeightedGraph):
    """Given a graph with nodes representing real
    points in the world, this function calculate the
    spatial dimension of the graph (in lat, lon angles)
    and the min,max lat, lon.
    """
    nodes = G.nodes(data=True)
    latitudes = [data["y"] for _, data in nodes]
    longitudes = [data["x"] for _, data in nodes]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    width = ox.distance.euclidean_dist_vec(min_lat, min_lon, min_lat, max_lon)
    height = ox.distance.euclidean_dist_vec(min_lat, min_lon, max_lat, min_lon)
    return width, height, min_lat, max_lat, min_lon, max_lon


def get_random_border_node(G: MultiDiGraph, seed=0):
    """Get a random node on the border of the graph G
    This defines border of a graph a node with only 1
    neighbor"""
    random.seed(seed)
    nodes = list(G.nodes)
    border_nodes = [node for node in nodes if len(list(G.neighbors(node))) == 1]
    return random.choice(border_nodes)


def create_highway_between_cities(G1: MultiDiGraph, G2: MultiDiGraph, n_seed=0):
    """Given 2 graphs representing two cities, this function create
    a link between them connecting with a highway two random nodes on
    the border"""

    _, _, min_lat, max_lat, min_lon, max_lon = graph_dimensions(G1)
    # print(f"dimension of city_1 lat {min_lat}-{max_lat}, lon {min_lon}-{max_lon}")
    _, _, min_lat, max_lat, min_lon, max_lon = graph_dimensions(G2)
    # print(f"dimension of city_2 lat {min_lat}-{max_lat}, lon {min_lon}-{max_lon}")

    # Get random border nodes
    node1 = get_random_border_node(G1, seed=n_seed)
    node2 = get_random_border_node(G2, seed=n_seed)

    G_combined = compose(G1, G2)

    # Add a highway (edge) between the two border nodes
    highway_length = ox.distance.euclidean_dist_vec(
        G1.nodes[node1]["y"],
        G1.nodes[node1]["x"],
        G2.nodes[node2]["y"],
        G2.nodes[node2]["x"],
    )
    G_combined.add_edge(
        node1,
        node2,
        length=(highway_length),
        highway="primary",
        maxspeed="130",
        oneway=False,
        reversed=False,
    )
    # add also in reverse direction
    G_combined.add_edge(
        node2,
        node1,
        length=(highway_length),
        highway="primary",
        maxspeed="130",
        oneway=False,
        reversed=False,
    )
    # print(f"edge added {G_combined.get_edge_data(node1, node2)}")

    return G_combined


def find_center_of_cities(G: WeightedGraph, n_clusters=2):
    """function that use cluster algoriths to find the centers
    of n city in a graph map"""
    nodes = pd.DataFrame(ox.graph_to_gdfs(G, edges=False))
    node_coords = nodes[["y", "x"]].values  # latitude and longitude pairs

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_coords)
    clusters = kmeans.labels_

    # Add cluster labels to nodes dataframe
    nodes["cluster"] = clusters

    # Calculate the centroid of each cluster
    centroids = []
    for cluster_label in nodes["cluster"].unique():
        cluster_nodes = nodes[nodes["cluster"] == cluster_label]
        centroid_lat = cluster_nodes["y"].mean()
        centroid_lon = cluster_nodes["x"].mean()
        centroids.append((centroid_lat, centroid_lon))

    return centroids


def ex3_get_expected_results() -> list[list[tuple[Path, int]]]:

    # The shortest path solutions for both UCS and Astar
    ny_path = [
        42436582,
        42446925,
        42448693,
        4597668041,
        42445867,
        42436575,
        42445879,
        42429876,
        42445885,
        42445888,
        42436746,
        42445404,
        42445896,
        42445899,
        42445903,
        42445365,
        42434948,
        42445908,
        42445909,
        42445910,
        42445651,
        42445914,
        42432438,
        561042200,
    ]
    eth_path = [
        131923881,
        131984636,
        1481512725,
        1481512717,
        1481512716,
        34466208,
        263683273,
        263683272,
    ]
    milan_path = [
        27653945,
        1550365269,
        1550365202,
        1550365234,
        1550365238,
        1550365233,
        249167226,
        1828995193,
        1476588224,
        3832161562,
        1514352941,
        1942874551,
        1942874549,
        1942874545,
        1942874539,
        1942874538,
        1942874527,
        1942874524,
        21226083,
        2075241344,
        480913573,
        28848583,
        28848589,
    ]

    # Each "expected result" contains the ideal path and the "trivial heuristic" count. For UCS, there is no
    # heuristic, so the evaluator will ignore this value. For Astar, the student's heuristic should be invoked
    # less often than the trivial heuristic -- the evaluator uses the trivial heuristic as a comparitor.
    # Passing the value 0 tells the evaluator to compute the trivial heuristic count using the student's Astar
    # implementation. Passing a non-zero value tells the evaluator to use this as the trivial heuristic count
    # (as is done in the evaluation server)
    expected_results = [
        [(ny_path, 0)],  # ucs ny
        [(ny_path, 0)],  # astar ny
        [(eth_path, 0)],  # ucs eth
        [(eth_path, 0)],  # astar eth
        [(milan_path, 0)],  # ucs milan
        [(milan_path, 0)],  # astar milan
    ]

    return expected_results


def ex3_compute_expected_results(test_values: list[TestValueEx3]) -> list[list[tuple[Path, int]]]:
    expected_results = []
    # loop over test cases
    for test in test_values:
        prob = test.problem
        # get graph and queries
        wG = prob.graph
        test_queries = prob.queries
        result = []
        # loop over queries
        for query in test_queries:
            try:
                path = astar_path(
                    G=wG._G,
                    source=query[0],
                    target=query[1],
                    heuristic=lambda v, u: 0,  # effectively Dijkstra
                    weight='travel_time'
                )
            except NetworkXNoPath:
                path = []
            result.append((path, 0))
        expected_results.append(result)
    return expected_results
