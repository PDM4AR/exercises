from pdm4ar.exercises.ex02.algo import BreadthFirst, DepthFirst, IterativeDeepening, WavefrontPlanner

graph_search_algo = {
    DepthFirst.__name__: DepthFirst,
    BreadthFirst.__name__: BreadthFirst,
    IterativeDeepening.__name__: IterativeDeepening,
}

wavefront_planner = {
    WavefrontPlanner.__name__: WavefrontPlanner,
}
