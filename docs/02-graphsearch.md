# Graph search :computer:

<table>
  <tr>
    <th><i>Prerequisites:</i></th><td><a href="./00-preliminaries.html" target="_top">Preliminaries</a></td><td><a href="./01-hello-world.html" target="_top">Hello-world</a></td>
  </tr>
</table>

## Breadth/Depth first search

As a first exercise we are going to implement Breadth/Depth first search and Iterative Deepening.

#### Graph structures

In this exercise we represent a directed graph via an *adjacency list*. Note that this is not the only possible
representation (e.g. adjacency matrices,...) but it is a very convenient one for graph search problems if the graph is
known a priori.

Given a generic type `X` for the nodes we associate to each node the list of successors, thus:

```python
AdjacencyList = Mapping[X, Set[X]]
```

#### Task

The task is to implement the _abstractmethod_ `search` for the different search techniques (`exercises/ex02/algo.py`).
Given a graph, a starting node, and a goal node the task is to return a sequence of states (_transitions_) from start to
goal.

```python
@abstractmethod
def search(self, graph: AdjacencyList, start: X, goal: X) -> Tuple[Path, OpenedNodes]:
    """
    :param graph: The given graph as an adjacency list
    :param start: The initial state (i.e. a node)
    :param goal: The goal state (i.e. a node)
    :return: tuple containing:
        1. The path from start to goal as a Sequence of states, [] if a path does not exist
        2. The list of opened nodes from the start until the last opened node
    """
    pass
```

The `search` method has to be implemented for 3 different algorithms: Breadth First Search, Depth First Search and Iterative Deepening. The method should return the Path from start to end node (empty list `[]` if not found) and the opened nodes `OpenedNodes` during the search.
When solving the graph search problem, the following conventions should hold:
* Nodes are represented by Integers.
* When a node is expanded, its neighbours are added to the queue in increasing order (from smaller to larger Int). Only the neighbours shall be sorted when added and not the nodes already in the queue.
Ex: The current queue is `Q = [0]`, if nodes to be added in DFS fashion are `{2, 1}` then the new queue will be `Q = [1, 2, 0]`. 
If a successor of the expanded node is already in the queue, it should not be added newly. I.e., given `Q = [0]`, with successors `{2, 1, 0}` then the new queue will be `Q = [1, 2, 0]`.

#### Test cases and performance criteria

The algorithms are going to be tested on different graphs, each containing randomly generated queries (start &
goal node).
You'll be able to test your algorithms on some test cases with given solution, both the `Path` and `OpenedNodes` will be compared to the solution. 
After running the exercise, you'll find reports in `out/[exercise]/` for each test case. There you'll be able to visualize the graphs, your output and the solution. These test cases aren't graded but serve as a guideline for how the exercise will be graded overall.

The final evaluation will combine 2 metrics lexicographically <accuracy,time>:
* **Accuracy**: The problem has been formulated to allow for 1 unique solution to both `Path` and `OpenedNodes`. 
A `Path`/`OpenedNodes` to be considered correct has to **fully** match the correct solution.
Averaging over the test cases we compute an accuracy metric as (# of correct paths)/(# of paths).Thus, accuracy will be in the interval [0, 1].
* **Solve time**: As your algorithms will be tested on graphs of increasing size, the efficiency of your code will be measured in terms of process time required.

#### Update your repo

In order to update your repo you can run the following command in the terminal:
```python
make update
```

##### Run the exercise

You can run your algorithms `locally` on the given test case with the following command:
```python
python3 src/pdm4ar/main.py -e 02
```

Alternatively you can open the project in a `VScode remote container`. The command to run the exercise is the same as above.

#### Food for thought

* Which of the graph search algorithms you implemented are better suited for different topologies of graphs?
* Does the presence of cycles in the graph affects any of the algorithms? If yes, why? Which modifications would you do to improve?
* Are the paths that you found the _shortest_ path?
