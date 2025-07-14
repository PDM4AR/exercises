from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import random
from random import seed, sample
from typing import Optional

import networkx as nx
from networkx import random_geometric_graph, DiGraph

from pdm4ar.exercises.ex02.structures import AdjacencyList, Path, OpenedNodes, Query, Grid


@dataclass
class GraphSearchProblem:
    queries: set[Query]
    graph_id: str
    graph: AdjacencyList


@dataclass
class GridSearchProblem(GraphSearchProblem):
    graph: Optional[AdjacencyList]
    grid: Grid


def networkx_2_adjacencylist(nxgraph: DiGraph) -> AdjacencyList:
    adj_list = dict()
    atlas = nxgraph.adj._atlas
    for n in atlas.keys():
        adj_list[n] = set(atlas[n].keys())
    return adj_list


def queries_from_adjacency(adj_list: AdjacencyList, n: int, n_seed=None) -> set[Query]:
    seed(n_seed)
    graph_queries = set()
    nodes = list(adj_list.keys())
    for _ in range(n):
        query_pair = sample(nodes, 2)
        graph_queries.add(tuple(query_pair))
    return graph_queries


def grid_to_adjacency_list(grid: Grid) -> AdjacencyList:
    """
    Converts a grid to an adjacency list.
    :param grid: A square matrix representing the grid.
    :return: An adjacency list representation of the graph.
    """
    adjacency_list: AdjacencyList = {}
    n = len(grid)
    for i in range(n):
        if len(grid[i]) != n:
            raise ValueError("Grid must be square.")

    def idx2id(r: int, c: int) -> int:
        return r * n + c + 1

    for r in range(n):
        for c in range(n):
            node_id = idx2id(r, c)
            neighbors = set()
            if grid[r][c] == 1:
                continue
            # Check all 4 possible directions (up, down, left, right)
            possible_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in possible_directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] != 1:
                    neighbor_id = idx2id(nr, nc)
                    neighbors.add(neighbor_id)
            adjacency_list[node_id] = set(sorted(neighbors))
    return adjacency_list


def generate_random_grid(n: int, d: float, seed: int = None) -> Grid:
    """
    Generate an n x n grid with density d of 1s, placed randomly.

    Parameters:
    - n (int): size of the grid (n x n)
    - d (float): density of 1s (between 0 and 1)
    - seed (int): random seed (optional)

    Returns:
    - Grid: grid with 0s and 1s
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    total_cells = n * n
    num_ones = int(round(total_cells * d))
    num_zeros = total_cells - num_ones

    # Create a flat list with the correct number of 1s and 0s
    values = [1] * num_ones + [0] * num_zeros
    random.shuffle(values)

    # Convert flat list into 2D grid
    grid = [values[i * n : (i + 1) * n] for i in range(n)]
    return grid


def generate_start_and_goal_from_grid(grid: Grid, seed: int = None) -> Query:
    """
    Generate a random start and goal position on a grid where both are on white squares (0s).

    Parameters:
    - grid (Grid): 2D grid with 0s (white) and 1s (black)
    - seed (int): optional random seed

    Returns:
    - Query: representing the start and goal positions as node IDs.
    """

    def idx2id(r: int, c: int) -> int:
        return r * n + c + 1

    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    n = len(grid)
    white_cells_id = [idx2id(i, j) for i in range(n) for j in range(n) if grid[i][j] == 0]

    if len(white_cells_id) < 2:
        raise ValueError("Not enough white cells to choose distinct start and goal positions.")

    start, goal = random.sample(white_cells_id, 2)
    return start, goal


def generate_queries_grid(grid: Grid, n: int, seed: int = None) -> set[Query]:
    """
    Generate a set of n unique (start, goal) queries on white squares.

    Parameters:
    - grid (Grid): 2D grid with 0s (white) and 1s (black)
    - n (int): number of query pairs to generate
    - seed (int): optional random seed

    Returns:
    - Set[Query]: set of (start, goal) position pairs
    """

    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    seeds = [random.randint(0, 10000) for _ in range(n)]
    queries = set()
    for seed in seeds:
        start, goal = generate_start_and_goal_from_grid(grid, seed)
        query: Query = (start, goal)
        if query not in queries:
            queries.add(query)

    if len(queries) < n:
        raise RuntimeError("Failed to generate enough unique queries.")
    return queries


def get_graph_search_problems(
    n_seed: int = None,
    height: int = 4,
    extra_test_graph_problems: list[GraphSearchProblem] = [],
) -> list[GraphSearchProblem]:
    graphsearch_prob = list()
    # test graph 1
    easy01_id = "easy01"
    easy01: AdjacencyList = {1: {2, 3}, 2: {3, 4}, 3: {4}, 4: {3}, 5: {6}, 6: {3}}
    easy01_queries = {(1, 4), (2, 6), (6, 1), (6, 6), (5, 4)}
    graphsearch_prob.append(GraphSearchProblem(graph=easy01, queries=easy01_queries, graph_id=easy01_id))

    # test graph 2
    size_g2 = 10
    graph02_id = "graph02"
    graph02_nx = random_geometric_graph(size_g2, 0.5, seed=9)
    graph02: AdjacencyList = networkx_2_adjacencylist(graph02_nx)
    graph02_queries = queries_from_adjacency(graph02, 3, n_seed)
    graphsearch_prob.append(GraphSearchProblem(graph=graph02, queries=graph02_queries, graph_id=graph02_id))

    # test grid 1
    grid_id = "grid01"
    # grid_test: AdjacencyList = {1: {2, 4}, 2: {1, 3}, 3: {2, 6}, 4: {1, 7}, 6: {3}, 7: {8, 4}, 8: {7}}
    grid01: Grid = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    grid01_queries = {(1, 8)}
    graphsearch_prob.append(GridSearchProblem(graph=None, grid=grid01, queries=grid01_queries, graph_id=grid_id))

    # test grid 2
    grid_id = "grid02"
    grid02: Grid = generate_random_grid(20, 0.25, 3)
    grid02_queries = generate_queries_grid(grid02, 3, 4)
    graphsearch_prob.append(GridSearchProblem(graph=None, grid=grid02, queries=grid02_queries, graph_id=grid_id))

    # # test graph 3
    # branching = 3
    # graph03_id = "graph03"
    # graph03_nx = nx.balanced_tree(branching, height)
    # graph03_nx = nx.bfs_tree(graph03_nx, 0)
    # graph03: AdjacencyList = networkx_2_adjacencylist(graph03_nx)
    # goals = sample(list(range(len(graph03))), 3)
    # graph03_queries = tuple(
    #     product(
    #         [
    #             0,
    #         ],
    #         goals,
    #     )
    # )
    # graphsearch_prob.append(
    #     GraphSearchProblem(graph=graph03, queries=graph03_queries, graph_id=graph03_id)
    # )

    for extra_problem in extra_test_graph_problems:
        graphsearch_prob.append(extra_problem)
    return graphsearch_prob


def ex2_get_expected_results() -> list[Optional[list[tuple[Path, OpenedNodes]]]]:
    expected_results = defaultdict(dict)

    # expected results graph 1 dfs
    expected_results[0] = [
        ([], [2, 3, 4]),
        ([6], [6]),
        ([], [6, 3, 4]),
        ([5, 6, 3, 4], [5, 6, 3, 4]),
        ([1, 2, 4], [1, 2, 4]),
    ]
    # expected results graph 1 bfs
    expected_results[1] = [
        ([], [2, 3, 4]),
        ([6], [6]),
        ([], [6, 3, 4]),
        ([5, 6, 3, 4], [5, 6, 3, 4]),
        ([1, 2, 4], [1, 2, 3, 4]),
    ]
    # expected results graph 1 id
    expected_results[2] = [
        ([], [2, 3, 4]),
        ([6], [6]),
        ([], [6, 3, 4]),
        ([5, 6, 3, 4], [5, 6, 3, 4]),
        ([1, 2, 4], [1, 2, 4]),
    ]

    # test graph 2 dfs
    expected_results[3] = [
        ([1, 2, 0, 6], [1, 2, 0, 6]),
        ([3, 6, 4], [3, 6, 0, 2, 1, 9, 5, 8, 4]),
        ([7, 0, 2], [7, 0, 2]),
    ]

    # test graph 2 bfs
    expected_results[4] = [
        ([1, 4, 6], [1, 2, 4, 5, 9, 0, 8, 6]),
        ([3, 6, 4], [3, 6, 7, 0, 4]),
        ([7, 0, 2], [7, 0, 3, 4, 6, 8, 2]),
    ]

    # test graph 2 id
    expected_results[5] = [
        ([1, 4, 6], [1, 2, 0, 8, 4, 6]),
        ([3, 6, 4], [3, 6, 0, 4]),
        ([7, 0, 2], [7, 0, 2]),
    ]

    # test grid 1 dfs
    expected_results[6] = [([1, 4, 7, 8], [1, 4, 7, 8])]

    # test grid 1 bfs
    expected_results[7] = [([1, 4, 7, 8], [1, 4, 7, 8])]

    # test grid 1 id
    expected_results[8] = [([1, 4, 7, 8], [1, 4, 7, 8])]

    # test grid 2 dfs
    expected_results[9] = [([], []), ([], []), ([], [])]

    # test grid 2 bfs
    expected_results[10] = [([], []), ([], []), ([], [])]

    # test grid 2 id
    expected_results[11] = [([], []), ([], []), ([], [])]

    # # test graph 3 bfs
    # expected_results[6] = [
    #     (
    #         [0, 3, 11],
    #         [
    #             0,
    #             1,
    #             4,
    #             13,
    #             40,
    #             41,
    #             42,
    #             14,
    #             43,
    #             44,
    #             45,
    #             15,
    #             46,
    #             47,
    #             48,
    #             5,
    #             16,
    #             49,
    #             50,
    #             51,
    #             17,
    #             52,
    #             53,
    #             54,
    #             18,
    #             55,
    #             56,
    #             57,
    #             6,
    #             19,
    #             58,
    #             59,
    #             60,
    #             20,
    #             61,
    #             62,
    #             63,
    #             21,
    #             64,
    #             65,
    #             66,
    #             2,
    #             7,
    #             22,
    #             67,
    #             68,
    #             69,
    #             23,
    #             70,
    #             71,
    #             72,
    #             24,
    #             73,
    #             74,
    #             75,
    #             8,
    #             25,
    #             76,
    #             77,
    #             78,
    #             26,
    #             79,
    #             80,
    #             81,
    #             27,
    #             82,
    #             83,
    #             84,
    #             9,
    #             28,
    #             85,
    #             86,
    #             87,
    #             29,
    #             88,
    #             89,
    #             90,
    #             30,
    #             91,
    #             92,
    #             93,
    #             3,
    #             10,
    #             31,
    #             94,
    #             95,
    #             96,
    #             32,
    #             97,
    #             98,
    #             99,
    #             33,
    #             100,
    #             101,
    #             102,
    #             11,
    #         ],
    #     ),
    #     (
    #         [0, 2, 8],
    #         [
    #             0,
    #             1,
    #             4,
    #             13,
    #             40,
    #             41,
    #             42,
    #             14,
    #             43,
    #             44,
    #             45,
    #             15,
    #             46,
    #             47,
    #             48,
    #             5,
    #             16,
    #             49,
    #             50,
    #             51,
    #             17,
    #             52,
    #             53,
    #             54,
    #             18,
    #             55,
    #             56,
    #             57,
    #             6,
    #             19,
    #             58,
    #             59,
    #             60,
    #             20,
    #             61,
    #             62,
    #             63,
    #             21,
    #             64,
    #             65,
    #             66,
    #             2,
    #             7,
    #             22,
    #             67,
    #             68,
    #             69,
    #             23,
    #             70,
    #             71,
    #             72,
    #             24,
    #             73,
    #             74,
    #             75,
    #             8,
    #         ],
    #     ),
    #     (
    #         [0, 2],
    #         [
    #             0,
    #             1,
    #             4,
    #             13,
    #             40,
    #             41,
    #             42,
    #             14,
    #             43,
    #             44,
    #             45,
    #             15,
    #             46,
    #             47,
    #             48,
    #             5,
    #             16,
    #             49,
    #             50,
    #             51,
    #             17,
    #             52,
    #             53,
    #             54,
    #             18,
    #             55,
    #             56,
    #             57,
    #             6,
    #             19,
    #             58,
    #             59,
    #             60,
    #             20,
    #             61,
    #             62,
    #             63,
    #             21,
    #             64,
    #             65,
    #             66,
    #             2,
    #         ],
    #     ),
    # ]  # , None, None]
    # # test graph 3 dfs
    # expected_results[7] = [
    #     ([0, 3, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    #     ([0, 2, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    #     ([0, 2], [0, 1, 2]),
    # ]
    # # test graph 3 id
    # expected_results[8] = [
    #     ([0, 3, 11], [0, 1, 4, 5, 6, 2, 7, 8, 9, 3, 10, 11]),
    #     ([0, 2, 8], [0, 1, 4, 5, 6, 2, 7, 8]),
    #     ([0, 2], [0, 1, 2]),
    # ]

    return expected_results
