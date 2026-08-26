from abc import abstractmethod, ABC

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes, DistanceDict


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, [] if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []

class WavefrontPlanner:

    def compute_cost_to_go(self, reverse_graph: AdjacencyList, goal: X) -> DistanceDict:
        """
        :param reverse_graph: The reversed graph as an adjacency list
        :param goal: The goal state (i.e. a node)
        :return: A dictionary having the graph nodes as keys, and the corresponding distances 
                 to the goal as the value (an int if reachable, inf otherwise)  
        """
        # todo implement here your solution
        return {}

    def extract_path(self, start: X, graph: AdjacencyList, cost_to_go: DistanceDict) -> Path:
        """
        :param start: The initial state (i.e. a node)
        :param graph: The given graph as an adjacency list
        :param cost_to_go: The cost to go from each node to the goal as a dictionary 
        :return: The path from start to goal as a Sequence of states, [] if a path does not exist
        """

        # todo implement here your solution   
        return []
