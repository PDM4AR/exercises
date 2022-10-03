import json
from abc import ABC, abstractmethod
from asyncio.log import logger
from dataclasses import asdict, dataclass, field
import traceback
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar

from reprep import Report
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from pdm4ar.exercises_def.structures_time import run_with_timer


class ExIn(ABC):
    @abstractmethod
    def str_id(self) -> str:
        """Implement this to return the string for the report identifier"""
        pass


ExInT = TypeVar("ExInT", bound=ExIn)
"""Generic type of the exercise input, has to implement methods of ExIn"""
ExOutT = TypeVar("ExOutT")
"""Generic type of the exercise output"""


@dataclass(frozen=True)
class PerformanceResults:
    def __str__(self):
        return json.dumps(asdict(self))


@dataclass
class Exercise(Generic[ExInT, ExOutT]):
    desc: str
    """Description of the exercise"""
    evaluation_fun: Callable[[ExInT, Optional[ExOutT]], Tuple[PerformanceResults, Report]]
    """
    Evaluation function for a test case. 
        Receives as input a test case and optionally an expected result.
        Returns a performance result object and a report 
    """
    perf_aggregator: Callable[[Sequence[PerformanceResults]], PerformanceResults]
    """Function combining the performance results over the multiple test cases """
    test_values: Sequence[ExInT] = field(default_factory=list)
    """A series of test cases for the exercise"""
    expected_results: Sequence[Optional[ExOutT]] = field(default_factory=list)
    """A series of expected results for the exercise, could be None if unknown"""
    test_case_timeout: float = 999
    """Timeout for each test case, in seconds?"""

    def __post_init__(self):
        if self.expected_results:
            assert len(self.expected_results) == len(
                    self.test_values), "Mismatch between expected values and test cases"
        # wrap evaluation function with timeout
        self.evaluation_fun = run_with_timer(self.evaluation_fun, self.test_case_timeout)


class ExerciseEvaluator(ABC):

    def __init__(self, exercise: Exercise) -> None:
        self.ex: Exercise = exercise

    def evaluate(self) -> Tuple[PerformanceResults, Report]:
        n_failed_test_cases: int = 0
        eval_outputs: List[Tuple[PerformanceResults, Report]] = []

        # evaluate each test case
        for i, test_input in enumerate(self.ex.test_values):
            try:
                expected_out = self.ex.expected_results[i] if self.ex.expected_results is not None else None
                eval_out = self.ex.evaluation_fun(test_input, expected_out)
            except Exception as e:
                n_failed_test_cases += 1
                print(f"Failed because of:\n {e.args} \n{''.join(traceback.format_tb(e.__traceback__))}")
                logger.info(f"Test case: \n{test_input} \nfailed because of:\n {e.args}")
                continue
            eval_outputs.append(eval_out)

        # combine all the evaluations
        r = Report("Evaluation")
        n_test_values = len(self.ex.test_values)
        r.text(
                "Evaluated", text=f"Succesfully evaluated {n_test_values - n_failed_test_cases}/{n_test_values}")

        overall_perf = self.ex.perf_aggregator(
                [out_res[0] for out_res in eval_outputs])
        r.text("OverallPerformance",
               text=f"{remove_escapes(debug_print(overall_perf))} ")

        # append all reports
        for i, out_res in enumerate(eval_outputs):
            r.add_child(out_res[1])
        return overall_perf, r
