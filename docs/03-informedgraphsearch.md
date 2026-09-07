# Informed Graph Search :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-helloworld.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Informed graph search

In this exercise we look at weighted graph and related search algorithms for finding the shortest path. 
Specifically, you are tasked with the implementation of three algorithms: Uniform Cost Search (UCS), bidirectional Uniform Cost Search (Bi-UCS), and A*.

### Graph structures

In this exercise we need to augment the `AdjacencyList` seen in <a href="./02-graphsearch.html" target="_top">Exercise
2</a>
to keep track of the weights on the edges. A simple extension is the following:

```python
@dataclass
class WeightedGraph:
    adj_list: AdjacencyList
    reverse_adj_list: AdjacencyList
    weights: Mapping[Tuple[X, X], float]
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

    def get_node_coordinates(self, u: X) -> Tuple[float, float]:
        """
        Method of class WeightedGraph:
        :param u: node id
        :return (x, y): coordinates (LON & LAT) of node u
        """
        return self._G.nodes[u][NodeAttribute.LONGITUDE], self._G.nodes[u][NodeAttribute.LATITUDE]
```


We will be using connectivity graphs of a few (famous) cities around the world; sometimes, these cities will also be connected to their nearest neighboring cities (you can find a clue on how this is done in the file 'exercises_def/ex03/data.py').
In order to properly implement your algorithms, you will need to get some property from the nodes (e.g., their position on the map).
You can access a nodes coordinate using the method `get_node_coordinates()`.

The edge weight between 2 nodes is given as the travel time required to go from a node to the other, and it is directly retrievable with the function `get_weight()`.

The graphs are directed. `adj_list` contains the successors of every node, while `reverse_adj_list` contains its predecessors. If the original graph contains an edge from `u` to `v`, a backward search can traverse from `v` to `u`, but the edge cost must still be retrieved with `get_weight(u, v)`.


### Task
Implement the following algorithms in `src/pdm4ar/exercises/ex03/algo.py`:

```python
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

    # ...provided code...

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # todo
        return 0
        
    def path(self, start: X, goal: X) -> Path:
        # todo
        return []
```

### Bidirectional Uniform Cost Search
Bidirectional UCS is a practical variant of UCS intended to improve efficiency over the standard algorithm by running a bidirectional search, with the potential (in appropriate graphs) to reduce the total number of explored nodes.

Bidirectional UCS runs the same search as UCS from both ends of the query. One
UCS starts at `start` and follows `adj_list` towards `goal`. The other starts
at `goal` and follows `reverse_adj_list` towards `start`. The backward search
must still use edge weights in their original direction: when it traverses
from `v` to a predecessor `u`, the corresponding original edge is `u -> v`.

You can therefore implement Bi-UCS by starting from your forward UCS and
adding the same search logic in the backward direction. Maintain the same
information that your UCS implementation needs separately for the two
directions.

At each iteration, compare the cost of the highest priority element in the forward queue (the minimum forward cost) with the one in the backward queue. Expand only the direction with the smaller minimum;
do not expand both queues in the same iteration. Apart from its direction,
each expansion follows the usual UCS logic.

In addition, maintain `mu`, the cost of the cheapest complete start-to-goal path found so far, initially infinity. Whenever discovering or improving the distance to a node `x`, check if `x` has already been discovered by the search in the opposite direction. If so, the two searches define a complete candidate path through `x`, with cost:

```text
distance_forward[x] + distance_backward[x]
```

If this value is smaller than `mu`, update `mu` and remember `x` as the best
meeting point so that the two path halves can eventually be joined. Note that the first
connection is not necessarily an optimal path, so the algorithm must not stop
as soon as the searches meet.

A safe termination condition (may only be used only after `mu` is finite) is:

```text
minimum_forward_queue_cost + minimum_backward_queue_cost >= mu
```

At this point, we can join the forward and backward parts of the best path associated with `mu` and return it (why is this a safe termination condition for an optimal path?).

Compared with your UCS code, the intended workflow is therefore:

1. use your UCS logic in the forward direction and in the backward
   direction;
2. expand only the direction whose queue currently has the smaller minimum
   cost;
3. update `mu` whenever the two searches connect; and
4. stop when the sum of the two queue minima is at least `mu`.

Bi-UCS is **not guaranteed to examine fewer edges than UCS on every query**.
Its benefit depends on the graph, the query, and how the two search frontiers
develop. Even a correct and efficient Bi-UCS implementation may have a search-efficiency ratio greater than `1.0` for an individual query.
For more information about Bidirectional UCS (sometimes referred to as Bi-directional Dijkstra): ([MIT 6.006 notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2008/resources/lec18/)), ([Princeton shortest-path notes](https://www.cs.princeton.edu/courses/archive/spr06/cos423/Handouts/EPP%20shortest%20path%20algorithms.pdf)). This material is only for reference and the conventions above might slightly differ. 

### A*
Unlike UCS, A* is an informed algorithm thus requires implementing a heuristic function. While worst time complexity is the same for UCS and A*, the use of an admissible heuristic often leads to a lower number of explored nodes to find the shortest path. If no path is found, your algorithms should return an empty list.

You are free to implement the `_INTERNAL_heuristic` function based on any metric of your choice (make sure it is admissible!).
There exist many distance metrics. Below is provided a visual representation of the most common.
![image](https://miro.medium.com/max/1220/0*WrVc0CpxoStXpACy.png)
[image reference](#https://miro.medium.com/max/1220/0*WrVc0CpxoStXpACy.png)

As mentioned, the edge weight between 2 nodes is given as travel time. There's a finite number (4) of speed regimes that can be followed along an edge, as represented in the class below.
You can access the speed value using the `.value` property of the struct, i.e. `HIGHWAY.value`.
```python
@unique
class TravelSpeed(float, Enum):
    HIGHWAY = 100.0 / 3.6
    SECONDARY = 70.0 / 3.6
    CITY = 50.0 / 3.6
    PEDESTRIAN = 5.0 / 3.6
```

You are **NOT allowed** to use any existing graph search function implemented in the libraries such as `networkx`.

In addition to path correctness, the evaluator measures search efficiency by counting edge-weight accesses made by your implementation. Calls through `WeightedGraph.get_weight()` are counted automatically; you do not need to maintain a counter yourself. The count reflects how many edges your search examines, including repeated examinations. This is a valuable proxy for the efficiency of the algorithm, as fewer weight calls correspond to a smaller number of explored nodes.

For every query, your count is divided by the number of edge-weight accesses made by a reference NetworkX UCS run on the same graph and query:

```text
search efficiency = your edge-weight accesses / reference UCS edge-weight accesses
```

The same ratio is reported for UCS, Bi-UCS and A*, both locally and during private evaluation. A value below 1 means that your implementation examined fewer edges than the reference UCS baseline. The aggregated search-efficiency score includes only Bi-UCS and A*. Only correct, non-trivial queries are included, and their counts are summed before the final ratio is computed. Always use the public weighted-graph interface and do not access the private NetworkX graph `_G`.

Some hints that may help you during the implementation of the algorithms: 
* The edge weight is the travel time between the 2 nodes, hence you should think about converting travel distance into travel time. 
Under which condition will the time metric be admissible?

* To obtain the distance between 2 coordinates, you may find useful the function `osmnx.distance.great_circle()`.

* For UCS, Bi-UCS and Astar, you may find Python's `heapq` module useful.

* You might want to organise your queue as `queue = [ (<priority>, <i = insertion order>, <node>, <cost-to-reach>, <parent_node>) ]`

In addition, we provide you with a script to allow you to increase and personalise your local test cases on the existing graphs. You can choose the `(start_node, goal_node)` tuples of int as query for your search algorithm without worrying they actually exist, as they will be checked and filtered. Moreover, a predefined function will generate existing random queries if you set a positive integer in the `n_random_queries` dict. Edit `src/pdm4ar/exercises_def/ex03/local_queries.py` in the apposite window:

```python
def get_local_queries(G: WeightedGraph, id: str) -> set[Query]:
    """
    Generate local queries for the given graph.
    Local queries are manually specified node pairs.
    Random queries are sampled from adjacent node pairs in the graph.
    """

    # === STUDENT-EDITABLE SECTION ===

    my_queries = {"ny": set(), 
                  "eth": set(), 
                  "milan": set()}                           # Replace set() with e.g.{(1, 2), (3, 4)}
    n_random_queries = {"ny": 0, "eth": 0, "milan": 0}      # Replace 0 with another int
    random_seed = None                                      # Set an integer if you want deterministic results

    # === END STUDENT-EDITABLE SECTION ===
```

### Test cases and performance criteria

The algorithms are going to be tested on different graphs, each containing randomly generated queries (start & goal node).
You will be able to test your algorithms on some test cases with given solution; the output `Path` will be compared to the solution. 
After running the exercise, you'll find reports in `out/[exercise]/` for each test case. There you'll be able to visualize the graphs, your output and the solution.
These test cases are not graded but serve as a guideline for how the exercise will be graded overall.

The final evaluation will combine 3 metrics lexicographically <number of solved cases, accuracy, efficiency>:
* **Accuracy**: UCS, Bi-UCS and A* will be evaluated. A `Path` to be considered correct has to **fully** match the correct solution. Averaging over the test cases we compute an accuracy metric as (# of correct paths)/(# of paths). Thus, accuracy will be in the interval [0, 1].
* **Efficiency**: Your efficiency score incorporates **both solve time and search efficiency**. The report shows your edge-weight accesses, the reference UCS count, and their ratio for each query. Only the Bi-UCS and A* ratios contribute to the aggregated search-efficiency score; the UCS ratio is shown for your convenience, to showcase your implementation's efficiency compared to the networkx baseline.

For reference, the TA’s solution achieves the following efficiency and solving times on the server:

| Metric              | Values      |
|---------------------|-------------|
| Search efficiency   |     0.7483  |
| Solve time [s]      |     0.0016  |

Use these numbers as a guideline to understand the order of magnitude of expected performance for a decently optimized solution.

### Useful remarks from last year Q&A
* Please, use the provided templates to implement your functions, without modifying the arguments and output numbers and types, unless stated otherwise.
* If not explicitly instructed otherwise, you may use anything included in the Docker environment to simplify your calculations. However, the evaluator blocks certain imports to prevent access to internal evaluation data or manipulation of recorded counts. Check the list of disallowed dependencies carefully, as using one will result in a score of zero.
* Since Uniform Cost Search is a special case of the A* algorithm, it is allowed to use A* implementation for UCS too, writing the code in the correct place and setting heuristic function = 0.
* **BE CAREFUL**: in the A* algorithm, the heuristic is summed to the cost-to-reach only for the ranking step in the queue, but you must **not** update the cost-to-reach with the heuristic estimate!
* For debugging, please keep in mind that your code has to work in all possible scenarios. Find them all!
