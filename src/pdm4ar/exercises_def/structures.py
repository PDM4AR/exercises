import json
import pprint

from tqdm import tqdm
from abc import ABC, abstractmethod
from pdm4ar.exercises_def import logger
from dataclasses import asdict, dataclass, field
from typing import Callable, Generic, List, Optional, Sequence, Tuple, TypeVar
import re
import os

from reprep import Report
from zuper_commons.text import remove_escapes

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


@dataclass(frozen=True)
class AggregatedPerformanceRes:
    n_test_cases: int
    """Number of total test cases"""
    n_completed_test_cases: int
    """Number of completed test cases during evaluation"""
    exceptions: List[str]
    """List of the raised exceptions names"""
    failed_test_id: List[int]
    """List of the failed test case ids"""
    perf_res: PerformanceResults
    """Performance results (already aggregated from the ones that did not fail)"""

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
    test_case_timeout: float = 60 * 3
    """Timeout for each test case, in seconds?"""

    def __post_init__(self):
        if self.expected_results:
            assert len(self.expected_results) == len(
                self.test_values
            ), "Mismatch between expected values and test cases"
        # wrap evaluation function with timeout
        self.evaluation_fun = run_with_timer(self.evaluation_fun, self.test_case_timeout)


class ExerciseEvaluator(ABC):

    def __init__(self, exercise: Exercise) -> None:
        self.ex: Exercise = exercise

    def evaluate(self) -> Tuple[AggregatedPerformanceRes, Report]:
        n_failed_test_cases: int = 0
        exceptions: List[str] = []
        eval_outputs: List[Tuple[PerformanceResults, Report]] = []

        # evaluate each test case
        failed_test_id: List[int] = []
        for i, test_input in enumerate(tqdm(self.ex.test_values)):
            expected_out = self.ex.expected_results[i] if self.ex.expected_results is not None else None
            eval_out = self.ex.evaluation_fun(test_input, expected_out)
            if isinstance(eval_out, Exception):
                n_failed_test_cases += 1
                failed_test_id.append(i + 1)
                exception_str = f"{eval_out.__class__.__name__}"
                exceptions.append(exception_str)
                logger.warn(f"Failed because of:\n {eval_out.args}")
                continue
            eval_outputs.append(eval_out)

        # combine all the evaluations
        r = Report("Evaluation")
        n_test_values = len(self.ex.test_values)
        n_completed_test_cases = n_test_values - n_failed_test_cases
        r.text("Evaluated", text=f"Test cases completion ratio:\n\t" f"({n_completed_test_cases}/{n_test_values})")

        agg_perf = self.ex.perf_aggregator([out_res[0] for out_res in eval_outputs])

        overall_perf = AggregatedPerformanceRes(
            n_completed_test_cases=n_completed_test_cases,
            n_test_cases=n_test_values,
            exceptions=exceptions,
            failed_test_id=failed_test_id,
            perf_res=agg_perf,
        )
        r.text("OverallPerformance", text=pprint.pformat(overall_perf))

        # append all reports
        for i, out_res in enumerate(eval_outputs):
            r.add_child(out_res[1])
        return overall_perf, r


def out_dir(exercise: str) -> str:
    repo_dir = __file__
    src_folder = "src"
    assert src_folder in repo_dir, repo_dir
    repo_dir = re.split(src_folder, repo_dir)[0]
    assert os.path.isdir(repo_dir)
    out_main = os.path.join(repo_dir, "out", exercise)
    return out_main


REPORT_CSS = """<style>
body { max-width: 1200px; margin: 0 auto; padding: 0 16px 32px; font-family: system-ui, sans-serif; }
h { font-family: inherit; font-size: 125%; margin: 0.6em 0 0.4em; }
.report-node { margin-bottom: 1.5em; padding: 0.4em 0.8em; }
.report-figure { background: none; }
pre.report-text { background: #f5f5f5; padding: 0.6em 0.9em; border-radius: 4px; }
img { max-width: 100%; height: auto; border: none; }
video { display: block; margin: 0.8em 0; }
</style>
"""


def embed_videos(report_file: str) -> None:
    """reprep only links MP4 files (hidden as data nodes): turn each link into a playable video.

    Also adds a light stylesheet for spacing and readability.
    """
    with open(report_file) as f:
        html = f.read()
    html = html.replace("<body>", REPORT_CSS + "<body>", 1)  # reprep writes no </head>
    html = re.sub(
        r'<p class="datanode"><a href="([^"]+\.mp4)">([^<]*)</a>:\s*video/mp4</p>',
        r'<video controls preload="metadata" style="width: 100%; max-width: 1100px" src="\1" title="\2"></video>',
        html,
    )
    with open(report_file, "w") as f:
        f.write(html)
