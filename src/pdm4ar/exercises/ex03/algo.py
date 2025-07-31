from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq  # you may find this helpful

from osmnx.distance import great_circle_vec

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

        ################################################################
        Path = [start]
        Queue = [[0.0, start]]  # [cost2Reach *float*, node *int*]
        heapq.heapify(Queue)
        cost2Reach = {start: 0.0}  # TIME !!!
        Parent = {start: 0}

        while Queue:
            pair = heapq.heappop(Queue)
            node = pair[1]  ## compatibility problem !! s must be in integer

            if node == goal:

                while node != start:  # found the goal (NOT EQUAL TO START) --> compute the path
                    Path.insert(1, node)
                    node = Parent[node]

                return Path

            else:
                for adjacent in self.graph.adj_list[node]:
                    newCost = cost2Reach[node] + self.graph.get_weight(node, adjacent)

                    if adjacent not in cost2Reach or newCost < cost2Reach[adjacent]:
                        cost2Reach[adjacent] = newCost
                        Parent[adjacent] = node

                        i = 0
                        max_len = len(Queue)
                        while (
                            i < max_len and Queue[i][1] != adjacent
                        ):  # look for adjacent in Queue --> pair after pair I have to compare the second item (node) to adjacent
                            i += 1

                        if i == max_len:  # not in Queue --> INSERT in Queue
                            heapq.heappush(Queue, [newCost, adjacent])

                        else:  # found --> UPDATE Cost2Reach in Queue
                            Queue[i] = [newCost, adjacent]
                            heapq.heapify(Queue)

        return []
        ################################################################
        # pass  REMEMBER TO UNCOMMENT THIS LINE FOR LAST PUSH


@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        # todo

        ####################################################################
        coord_u = self.graph.get_node_coordinates(u)  # long-lat
        lat_u = coord_u[1]
        long_u = coord_u[0]

        coord_v = self.graph.get_node_coordinates(v)
        lat_v = coord_v[1]
        long_v = coord_v[0]

        distance = great_circle_vec(lat_u, long_u, lat_v, long_v, 6370) * 1000  # meters

        if distance < 30:
            heuristic = distance / TravelSpeed.PEDESTRIAN.value

        elif distance < 400:
            heuristic = distance / TravelSpeed.CITY.value

        elif distance < 880:
            heuristic = distance / TravelSpeed.SECONDARY.value

        else:
            heuristic = distance / TravelSpeed.HIGHWAY.value

        heuristic = 3 * heuristic

        return heuristic
        ####################################################################

        return 0

    def path(self, start: X, goal: X) -> Path:
        # todo

        #################################################################
        Path = [start]
        Queue = [[0.0, start]]  # [cost2Reach *float*, node *int*]
        heapq.heapify(Queue)
        cost2Reach = {start: 0.0}
        Parent = {start: 0}

        while Queue:
            pair = heapq.heappop(Queue)
            node = pair[1]

            if node == goal:

                while node != start:  # found the goal (NOT EQUAL TO START) --> compute the path
                    Path.insert(1, node)
                    node = Parent[node]

                return Path

            else:
                for adjacent in self.graph.adj_list[node]:
                    heuristic = self.heuristic(adjacent, goal) / 3
                    newCost = cost2Reach[node] + self.graph.get_weight(node, adjacent)

                    if adjacent not in cost2Reach or newCost < cost2Reach[adjacent]:
                        cost2Reach[adjacent] = newCost  # NO HEURISTIC
                        Parent[adjacent] = node

                        i = 0
                        max_len = len(Queue)
                        while (
                            i < max_len and Queue[i][1] != adjacent
                        ):  # look for adjacent in Queue --> pair after pair I have to compare the second item (node) to adjacent
                            i += 1

                        if i == max_len:  # not in Queue --> INSERT in Queue
                            heapq.heappush(Queue, [newCost + heuristic, adjacent])  # here sum heuristic (IN QUEUE)

                        else:  # found --> UPDATE Cost2Reach in Queue
                            Queue[i] = [newCost + heuristic, adjacent]  # here sum heuristic (IN QUEUE)
                            heapq.heapify(Queue)

        return []
        #################################################################

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
