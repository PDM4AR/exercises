import warnings
from pdm4ar.exercises_def.ex02.data import queries_from_adjacency
from pdm4ar.exercises.ex02.structures import Query
from pdm4ar.exercises.ex03.structures import WeightedGraph


def get_local_queries(G: WeightedGraph, id: str) -> set[Query]:
    """
    Generate local queries for the given graph.
    Local queries are manually specified node pairs.
    Random queries are sampled from adjacent node pairs in the graph.
    """

    # === STUDENT-EDITABLE SECTION ===
    my_queries = {"ny": set(), 
                  "eth": {(131923881, 131923881)}, 
                  # DEBUGGING: for "eth", the query (89010597, 1552985925) has no solution
                  "milan": set()}                   # e.g., {(1, 2), (3, 4)}
    n_random_queries = {"ny": 1, "eth": 1, "milan": 1}
    random_seed = None  # Set an integer if you want deterministic results
    # === END STUDENT-EDITABLE SECTION ===

    if id not in my_queries:
        warnings.warn(f"Unknown graph id: {id}")
        return set()

    local_queries = check_local_queries(my_queries[id], id, G) if my_queries[id] else set()
    n_rnd = n_random_queries[id]
    random_queries = queries_from_adjacency(G.adj_list, n=n_rnd, n_seed=random_seed)

    return local_queries | random_queries


def check_local_queries(local_queries: set[Query], id: str, G: WeightedGraph) -> set[Query]:
    """
    Filter queries to include only those where both nodes exist in the graph.
    """
    graph = G._G  # underlying nx.MultiDiGraph
    valid_queries = {q for q in local_queries if q[0] in graph and q[1] in graph}
    invalid_queries = local_queries - valid_queries
    for q in invalid_queries:
        print(f"Query {q} does not exist in the graph {id}. Removing it.")
    return valid_queries
