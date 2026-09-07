from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq  # you may find this helpful

from osmnx.distance import great_circle

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        """
        Given the start and goal nodes, returns an ordered list of nodes from self.graph
        that make up the path between them, or an empty list if no path exists.
        """
        # Abstract function. Nothing to do here.
        pass


@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # todo
        pass


@dataclass
class BidirectionalUniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # todo
        pass


@dataclass
class Astar(InformedGraphSearch):
    def heuristic(self, u: X, v: X) -> float:
        return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        # todo
        return 0

    def path(self, start: X, goal: X) -> Path:
        # todo
        return []


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
