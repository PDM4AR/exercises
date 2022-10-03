from dataclasses import dataclass
from enum import Enum, unique
from typing import Tuple, Mapping, Optional, Any

from networkx import MultiDiGraph
from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import AdjacencyList, X


class EdgeNotFound(Exception):
    pass


class NodePropertyNotFound(Exception):
    pass


class WrongHeuristic(Exception):
    pass

@unique
class Heuristic(Enum):
    MANHATTAN = 0
    EUCLIDEAN = 1
    CHEBYSHEV = 2
    INADMISSIBLE = 3

@unique
class NodeAttribute(str, Enum):
    LONGITUDE = 'x'
    LATITUDE = 'y'

@unique
class TravelSpeed(float, Enum):
    HIGHWAY = 100.0 / 3.6
    SECONDARY = 70.0 / 3.6
    CITY = 50.0 / 3.6
    PEDESTRIAN = 5.0 / 3.6


@dataclass
class WeightedGraph:
    
    def __init__(self, adj_list, weights, _G) -> None:
        # init graph
        self.adj_list: AdjacencyList = adj_list
        self.weights: Mapping[Tuple[X, X], float] = weights
        self._G: MultiDiGraph = _G

    def get_weight(self, u: X, v: X) -> Optional[float]:
        """
        :param u: The "from" of the edge
        :param v: The "to" of the edge
        :return: The weight associated to the edge, raises an Exception if the edge does not exist
        """
        try:
            return self.weights[(u, v)]
        except KeyError:
            raise EdgeNotFound(f"Cannot find weight for edge: {(u, v)}")

    def __get_node_attribute(self, node_id: X, attribute: NodeAttribute) -> Any:
        """
        Private method of class WeightedGraph
        :param node_id: The node id
        :param attribute: The node attribute name
        :return: The corresponding value
        """
        return self._G.nodes[node_id][attribute]

    def __get_node_coordinates(self, u: X) -> Tuple[float]:
        # todo
        return ()

    def heuristic_manhattan(self, u:X, v:X) -> float:
        # todo
        pass
    
    def heuristic_euclidean(self, u:X, v:X) -> float:
        # todo
        pass
    
    def heuristic_chebyshev(self, u:X, v:X) -> float:
        # todo
        pass
    
    def heuristic_inadmissible(self, u:X, v:X) -> float:
        # todo
        pass

    def get_heuristic(self, u: X, goal: X, heuristic: Optional[Heuristic]) -> float:
        """
        :param u: The current node
        :param goal: The goal node of the query
        :return: The associated heuristic cost to go to the goal node
        """

        if heuristic is None:
            return 0.0

        # todo 
        pass


