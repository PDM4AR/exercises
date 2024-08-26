from dataclasses import dataclass
from enum import Enum, unique
from typing import Mapping, Optional, Any

from networkx import MultiDiGraph

from pdm4ar.exercises.ex02.structures import AdjacencyList, X


class EdgeNotFound(Exception):
    pass


class NodePropertyNotFound(Exception):
    pass


@unique
class NodeAttribute(str, Enum):
    LONGITUDE = "x"
    LATITUDE = "y"


@unique
class TravelSpeed(float, Enum):
    HIGHWAY = 100.0 / 3.6
    SECONDARY = 70.0 / 3.6
    CITY = 50.0 / 3.6
    PEDESTRIAN = 5.0 / 3.6


@dataclass
class WeightedGraph:
    adj_list: AdjacencyList
    weights: Mapping[tuple[X, X], float]
    _G: MultiDiGraph

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

    def _get_node_attribute(self, node_id: X, attribute: NodeAttribute) -> Any:
        """
        Private method of class WeightedGraph
        :param node_id: The node id
        :param attribute: The node attribute name
        :return: The corresponding value
        """
        return self._G.nodes[node_id][attribute]

    def get_node_coordinates(self, u: X) -> tuple[float, float]:
        """
        Method of class WeightedGraph:
        :param u: node id
        :return (x, y): coordinates (LON & LAT) of node u
        """
        return (
            self._G.nodes[u][NodeAttribute.LONGITUDE],
            self._G.nodes[u][NodeAttribute.LATITUDE],
        )
