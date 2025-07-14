from typing import Optional, TypeVar, Set, Mapping, Tuple, List

X = TypeVar("X")
"""A generic type for nodes in a graph."""

AdjacencyList = Mapping[X, Set[X]]
"""An adjacency list from node to a set of nodes."""

Query = Tuple[X, X]
"""A query as a tuple of start and goal nodes."""

Path = Optional[List[X]]
"""A path as a list of nodes."""

OpenedNodes = Optional[List[X]]
"""Also the opened nodes is a list of nodes"""

Grid = List[List[int]]
"""A grid represented as a list of lists of integers, where 0 is free space and 1 is an obstacle."""
