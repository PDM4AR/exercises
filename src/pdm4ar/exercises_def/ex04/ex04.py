from dataclasses import dataclass
from typing import Any, Type, Sequence

from matplotlib import pyplot as plt
from reprep import Report, MIME_PDF

from pdm4ar.exercises_def.ex04.map import map2image
from pdm4ar.exercises.ex04.mdp import GridMdpSolver
from pdm4ar.exercises.ex04.policy_iteration import PolicyIteration
from pdm4ar.exercises.ex04.value_iteration import ValueIteration
from pdm4ar.exercises_def import Exercise, ExIn
from pdm4ar.exercises_def.ex04.data import get_test_grids
from pdm4ar.exercises_def.ex04.utils import action2arrow, head_width
from pdm4ar.exercises_def.structures import PerformanceResults


@dataclass
class TestValueEx4(ExIn):
    algo: Type[GridMdpSolver]

    def str_id(self) -> str:
        return str(self.algo.__name__)

@dataclass(frozen=True)
class Ex04PerformanceResult(PerformanceResults):
    accuracy_mse: float # mean squared error between ground truth and solution
    solve_time: float

    def __post__init__(self):
        assert self.accuracy <= 1, self.accuracy
        assert self.solve_time >= 0, self.solve_time


def ex4_evaluation(ex_in: TestValueEx4, ex_out=None) -> Report:

    solver: GridMdpSolver = ex_in.algo()
    algo_name = solver.__class__.__name__
    r = Report(f"Ex4-{algo_name}")
    
    for k, grid_mdp in enumerate(get_test_grids()):
        
        value_func, policy = solver.solve(grid_mdp)

        MAP_SHAPE = grid_mdp.grid.shape
        font_size = 3 if MAP_SHAPE[0] > 15 else 6

        rfig = r.figure(cols=2)
        with rfig.plot(nid=f"{algo_name}-value-{k}", mime=MIME_PDF, figsize=None) as _:
            ax = plt.gca()
            ax.imshow(value_func, aspect="equal")
            ax.tick_params(axis="both", labelsize=font_size + 3)
            for i in range(MAP_SHAPE[0]):
                for j in range(MAP_SHAPE[1]):
                    ax.text(j, i, f"{value_func[i, j]:.1f}", size=font_size, ha="center", va="center", color="k")

        map_c = map2image(grid_mdp.grid)
        with rfig.plot(nid=f"{algo_name}-policy-{k}", mime=MIME_PDF, figsize=None) as _:
            ax = plt.gca()
            ax.imshow(map_c, aspect="equal")
            ax.tick_params(axis="both", labelsize=font_size + 3)
            for i in range(MAP_SHAPE[0]):
                for j in range(MAP_SHAPE[1]):
                    arrow = action2arrow[policy[i, j]]
                    ax.arrow(j, i, arrow[1], arrow[0], head_width=head_width, color="k")
    
    return Ex04PerformanceResult(accuracy_mse=0.0, solve_time=0.0), r # TODO: add performance


def ex4_perf_aggregator(perf: Sequence[Ex04PerformanceResult]) -> Ex04PerformanceResult:
    return Ex04PerformanceResult(accuracy_mse=0.0, solve_time=0.0) # TODO: add function


def get_exercise4() -> Exercise:
    test_values = [TestValueEx4(ValueIteration), TestValueEx4(PolicyIteration)]

    expected_results = None # TODO: add ground truth

    return Exercise[TestValueEx4, Any](
            desc='This exercise is about graph search',
            evaluation_fun=ex4_evaluation,
            perf_aggregator=ex4_perf_aggregator,
            test_values=test_values,
            expected_results=expected_results,
    )
