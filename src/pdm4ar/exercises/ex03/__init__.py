from .algo import *

informed_graph_search_algo = {
    UniformCostSearch.__name__: UniformCostSearch,
    BidirectionalUniformCostSearch.__name__: BidirectionalUniformCostSearch,
    Astar.__name__: Astar,
}
