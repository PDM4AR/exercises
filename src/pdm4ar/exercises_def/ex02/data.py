from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from random import seed, sample
from typing import Set, List, Tuple, Optional

import networkx as nx
from networkx import random_geometric_graph, DiGraph

from pdm4ar.exercises.ex02.structures import AdjacencyList, Path, OpenedNodes, Query


@dataclass
class GraphSearchProblem:
    graph: AdjacencyList
    queries: Set[Query]
    graph_id: str


def networkx_2_adjacencylist(nxgraph: DiGraph) -> AdjacencyList:
    adj_list = dict()
    atlas = nxgraph.adj._atlas
    for n in atlas.keys():
        adj_list[n] = set(atlas[n].keys())
    return adj_list


def queries_from_adjacency(adj_list: AdjacencyList, n: int, n_seed=None) -> Set[Query]:
    seed(n_seed)
    graph_queries = set()
    nodes = list(adj_list.keys())
    for _ in range(n):
        query_pair = sample(nodes, 2)
        graph_queries.add(tuple(query_pair))
    return graph_queries



def get_graph_search_problems(n_seed: int=None, height: int=4, extra_test_graph_problems: List[GraphSearchProblem]=[]) -> List[GraphSearchProblem]:
    graphsearch_prob = list()
    # test graph 1
    easy01_id = 'easy01'
    easy01: AdjacencyList = {1: {2, 3}, 2: {3, 4}, 3: {4}, 4: {3}, 5: {6}, 6: {3}}
    easy01_queries = {(1, 4), (2, 6), (6, 1), (6, 6), (5, 4)}
    graphsearch_prob.append(GraphSearchProblem(graph=easy01, queries=easy01_queries, graph_id=easy01_id))
    
    
    # test graph 2
    size_g2 = 10
    graph02_id = 'graph02'
    graph02_nx = random_geometric_graph(size_g2, 0.5, seed=2)
    graph02: AdjacencyList = networkx_2_adjacencylist(graph02_nx)
    graph02_queries = queries_from_adjacency(graph02, 3, n_seed)

    graphsearch_prob.append(GraphSearchProblem(graph=graph02, queries=graph02_queries, graph_id=graph02_id))

    # test graph three
    branching = 3
    graph03_id = 'graph03'
    graph03_nx = nx.balanced_tree(branching, height)
    graph03_nx = nx.bfs_tree(graph03_nx, 0)
    graph03: AdjacencyList = networkx_2_adjacencylist(graph03_nx)
    goals = sample(list(range(len(graph03))), 3)
    graph03_queries = tuple(
        product(
            [
                0,
            ],
            goals,
        )
    )
    graphsearch_prob.append(GraphSearchProblem(graph=graph03, queries=graph03_queries, graph_id=graph03_id))

    if evaluation:
        # test graph 4
        size_g4 = 200
        graph04_id = 'graph04'
        graph04_nx = random_geometric_graph(size_g4, 0.125, seed=2)
        graph04: AdjacencyList = networkx_2_adjacencylist(graph04_nx)
        graph04_queries = queries_from_adjacency(graph04, 1, n_seed)

        graphsearch_prob.append(GraphSearchProblem(graph=graph04, queries=graph04_queries, graph_id=graph04_id))

    return graphsearch_prob

def ex2_get_expected_results() -> List[Optional[List[Tuple[Path, OpenedNodes]]]]:
    expected_results = defaultdict(dict)


    # expected results graph 1 dfs
    expected_results[0] = [([], [2, 3, 4]),
                            ([6], [6]),
                            ([], [6, 3, 4]),
                            ([5, 6, 3, 4], [5, 6, 3, 4]),
                            ([1, 2, 4], [1, 2, 4])]
    # expected results graph 1 bfs
    expected_results[1] = [([], [2, 3, 4]),
                            ([6], [6]),
                            ([], [6, 3, 4]),
                            ([5, 6, 3, 4], [5, 6, 3, 4]),
                            ([1, 2, 4], [1, 2, 3, 4])]
    # expected results graph 1 id
    expected_results[2] = [([], [2, 3, 4]),
                            ([6], [6]),
                            ([], [6, 3, 4]),
                            ([5, 6, 3, 4], [5, 6, 3, 4]),
                            ([1, 2, 4], [1, 2, 4])]

    # test graph 2 dfs
    expected_results[3] = [([1, 6], [1, 6]),
                            ([3, 4], [3, 2, 0, 7, 4]),
                            ([7, 2], [7, 0, 8, 3, 5, 6, 1, 9, 2])]

    # test graph 2 bfs
    expected_results[4] = [([1, 6], [1, 6]),
                            ([3, 4], [3, 2, 4]),
                            ([7, 2], [7, 0, 2])]

    # test graph 2 id
    expected_results[5] = [([1, 6], [1, 6]),
                            ([3, 4], [3, 2, 4]),
                            ([7, 2], [7, 0, 2])]

    # test graph 3 bfs
    expected_results[6] = [( [0, 3, 11] , [0, 1, 4, 13, 40, 41, 42, 14, 43, 44, 45, 15, 46, 47, 48, 5, 16, 49, 50, 51, 17, 52, 53, 54, 18, 55, 56, 57, 6, 19, 58, 59, 60, 20, 61, 62, 63, 21, 64, 65, 66, 2, 7, 22, 67, 68, 69, 23, 70, 71, 72, 24, 73, 74, 75, 8, 25, 76, 77, 78, 26, 79, 80, 81, 27, 82, 83, 84, 9, 28, 85, 86, 87, 29, 88, 89, 90, 30, 91, 92, 93, 3, 10, 31, 94, 95, 96, 32, 97, 98, 99, 33, 100, 101, 102, 11] ),
                            ( [0, 2, 8] , [0, 1, 4, 13, 40, 41, 42, 14, 43, 44, 45, 15, 46, 47, 48, 5, 16, 49, 50, 51, 17, 52, 53, 54, 18, 55, 56, 57, 6, 19, 58, 59, 60, 20, 61, 62, 63, 21, 64, 65, 66, 2, 7, 22, 67, 68, 69, 23, 70, 71, 72, 24, 73, 74, 75, 8] ),
                            ( [0, 2] , [0, 1, 4, 13, 40, 41, 42, 14, 43, 44, 45, 15, 46, 47, 48, 5, 16, 49, 50, 51, 17, 52, 53, 54, 18, 55, 56, 57, 6, 19, 58, 59, 60, 20, 61, 62, 63, 21, 64, 65, 66, 2] )]#, None, None]
    # test graph 3 dfs
    expected_results[7] = [( [0, 3, 11] , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] ),
                            ( [0, 2, 8] , [0, 1, 2, 3, 4, 5, 6, 7, 8] ),
                            ( [0, 2] , [0, 1, 2] )]
    # test graph 3 id
    expected_results[8] = [( [0, 3, 11] , [0, 1, 4, 5, 6, 2, 7, 8, 9, 3, 10, 11] ),
                            ( [0, 2, 8] , [0, 1, 4, 5, 6, 2, 7, 8] ),
                            ( [0, 2] , [0, 1, 2] )]

    return expected_results
    