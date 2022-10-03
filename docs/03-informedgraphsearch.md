# Informed Graph Search :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Informed graph search

### Graph structures

In this exercise we need to augment the `AdjacencyList` seen in <a href="./02-graphsearch.html" target="_top">Exercise
2</a>
to keep track of the weights of the edges. A simple extension is the following:

```python
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

    def __get_node_coordinates(self, u: X) -> List[float]:
        # todo
        return ()

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
```

To properly implement a heuristic, we will need to get some property from the nodes (e.g., their position on the map). You'll have to implement the private method `__get_node_coordinates` in order to obtain the position (x, y) of each node.
For that you can use the private method `__get_node_attribute`. For example, in this exercises the graphs will be maps; a custom struct `NodeAttribute` is provided such that each node attribute can be retrieved within the class `WeightedGraph` as follows:

```python
node: X  # the node id
lon = self.__get_node_attribute(node, NodeAttribute.LONGITUDE)
lat = self.__get_node_attribute(node, NodeAttribute.LATITUDE)
```

The edge weight between 2 nodes is given as the travel time required to go from a node to the other and it will be directly retrievable from the function `get_weight`.

In order to implement the function`get_heuristic` it is sufficient to know that there are in total 4 possible travel speeds. You can access the speed floating value using the `.value` property of the struct.
```python
@unique
class TravelSpeed(float, Enum):
    HIGHWAY = 100.0 / 3.6
    SECONDARY = 70.0 / 3.6
    CITY = 50.0 / 3.6
    PEDESTRIAN = 5.0 / 3.6
```

You will have to implement 4 different metrics for your heuristic function. 
```python
@unique
class Heuristic(Enum):
    MANHATTAN = 0
    EUCLIDEAN = 1
    CHEBYSHEV = 2
    INADMISSIBLE = 3
```
You can easily google what each of the first 3 metrics represents. Below is provided a visual representation of these distance metrics.
![image](https://miro.medium.com/max/1220/0*WrVc0CpxoStXpACy.png)

As for the `INADMISSIBLE` heuristic, you are free to explore and see the effects on the optimality of your search algorithms.

(HINT 1) As already specified, the edge weight is the travel time between the 2 nodes, hence you should think about converting travel distance into travel time. Moreover, which of the distance metrics will provide an admissible heuristic? Under which conditions?
(HINT 2) To obtain the distance between 2 coordinates, you may find useful the function `osmnx.distance.great_circle_vec`.


### Task

Implement the following functions in `src/pdm4ar/exercises/ex03/structures.py`:

```python
def __get_node_coordinates(self, u: X) -> List[float]:
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


```

Implement the following algorithms in `src/pdm4ar/exercises/ex03/algo.py`:

```python
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
```

Note that the type of heuristic is an input argument to the `search` function. For `UniformCostSearch` it will be `None`.


#### Test cases and performance criteria

The algorithms are going to be tested on different graphs, each containing randomly generated queries (start &
goal node).
You'll be able to test your algorithms on some test cases with given solution, the outputted `Path` will be compared to the solution. 
After running the exercise, you'll find reports in `out/[exercise]/` for each test case. There you'll be able to visualize the graphs, your output and the solution. These test cases aren't graded but serve as a guideline for how the exercise will be graded overall.

The final evaluation will combine 2 metrics lexicographically <accuracy,time>:
* **Accuracy**: All 3 algorithms will be evaluated, however only on the admissible heuristics. A `Path` to be considered correct has to **fully** match the correct solution. Averaging over the test cases we compute an accuracy metric as (# of correct paths)/(# of paths). Thus, accuracy will be in the interval [0, 1].
* **Solve time**: As your algorithms will be tested on graphs of increasing size, the efficiency of your code will be measured in terms of process time required.

#### Update your repo



###### Run the exercise



#### Food for thoughts

* Which of the methods above is supposed to always find the shortest path?
* What are valid heuristic you can think of for the A* algorithm? Given the different topology of the three cities, do you expect some to work better on specific cities?
