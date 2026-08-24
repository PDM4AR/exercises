"""Evaluator-owned weight counter for Exercise 3.

This module is outside the student-mounted ``pdm4ar.exercises.ex03``
directory.  Student algorithms receive a restricted graph view and can access
edge weights only through ``get_weight``.
"""

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Optional

from pdm4ar.exercises.ex02.structures import AdjacencyList, X
from pdm4ar.exercises.ex03.structures import EdgeNotFound, WeightedGraph


class WeightLookupCounter(Mapping[tuple[X, X], float]):
    """Read-only weight mapping whose count is maintained by the evaluator."""

    __slots__ = ("__weights", "__count")

    def __init__(self, weights: Mapping[tuple[X, X], float]):
        self.__weights = weights
        self.__count = 0

    @property
    def count(self) -> int:
        return self.__count

    def __getitem__(self, edge: tuple[X, X]) -> float:
        self.__count += 1
        return self.__weights[edge]

    def __iter__(self) -> Iterator[tuple[X, X]]:
        return iter(self.__weights)

    def __len__(self) -> int:
        return len(self.__weights)


class InstrumentedWeightedGraph:
    """Restricted graph interface passed to student search implementations.

    The original weight mapping and NetworkX graph are deliberately not
    exposed. Coordinates are copied so this view does not retain the original
    ``WeightedGraph`` instance.
    """

    __slots__ = ("adj_list", "reverse_adj_list", "__coordinates", "__weights")

    adj_list: AdjacencyList
    reverse_adj_list: AdjacencyList

    def __init__(self, graph: WeightedGraph, weights: WeightLookupCounter):
        self.adj_list = graph.adj_list
        self.reverse_adj_list = graph.reverse_adj_list
        self.__coordinates = MappingProxyType(
            {node: graph.get_node_coordinates(node) for node in graph.adj_list}
        )
        self.__weights = weights

    def get_weight(self, u: X, v: X) -> Optional[float]:
        try:
            return self.__weights[(u, v)]
        except KeyError as exc:
            raise EdgeNotFound(f"Cannot find weight for edge: {(u, v)}") from exc

    def get_node_coordinates(self, u: X) -> tuple[float, float]:
        return self.__coordinates[u]


def instrument_weight_lookups(
    graph: WeightedGraph,
) -> tuple[InstrumentedWeightedGraph, WeightLookupCounter]:
    """Create the student graph view and its evaluator-owned counter."""

    counter = WeightLookupCounter(graph.weights)
    return InstrumentedWeightedGraph(graph, counter), counter
