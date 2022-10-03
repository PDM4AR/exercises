from abc import ABC, abstractmethod
from typing import Optional, List

from pdm4ar.exercises.ex02.structures import X
from pdm4ar.exercises.ex03.structures import WeightedGraph, Heuristic


class InformedGraphSearch(ABC):
    @abstractmethod
    def path(self, graph: WeightedGraph, start: X, goal: X, heuristic: Optional[Heuristic]) -> Optional[List[X]]:
        # need to introduce weights!
        pass


class UniformCostSearch(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X, heuristic: Optional[Heuristic]) -> Optional[List[X]]:
        # todo
        pass


class GreedyBestFirst(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X, heuristic: Optional[Heuristic]) -> Optional[List[X]]:
        # todo
        pass


class Astar(InformedGraphSearch):
    def path(self, graph: WeightedGraph, start: X, goal: X, heuristic: Optional[Heuristic]) -> Optional[List[X]]:
        # todo
        pass


def compute_path_cost(wG: WeightedGraph, path: List[X]):
    """A utility function to compute the cumulative cost along a path"""
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
