from dataclasses import dataclass
from random import random
from typing import List, Optional, Tuple

import numpy as np
from pdm4ar.exercises import ComparisonOutcome, compare_lexicographic
from pdm4ar.exercises_def import Exercise, PerformanceResults
from reprep import Report
from zuper_commons.text import remove_escapes


@dataclass(frozen=True)
class LexiPerformance(PerformanceResults):
    accuracy: float
    """Percentage of correct comparisons"""

    def __post_init__(self):
        assert 0 <= self.accuracy <= 1


def exercise1_eval(
    algo_in: List[Tuple[np.ndarray]],
    expected: List[Optional[ComparisonOutcome]],
) -> Tuple["LexiPerformance", Report]:
    """Evaluation function for exercise 1. Note that here one test case is already a bunch of single queries"""

    r = Report("TestCase-Ex1")
    correct_answers: int = 0
    test_results = []

    for i, value in enumerate(algo_in):
        res = compare_lexicographic(tuple(value[0]), tuple(value[1]))
        test_results.append(res)
        correct_out = expected[i] if expected is not None else None
        if correct_out is not None and correct_out == res:
            correct_answers += 1

        va, vb = value
        msg = f"Input:\na:\t{va}\tb:\t{vb}" f"\nOutput:\n\t{res}" f"\nExpectedOutput:\n\t{correct_out}"
        r.text(f"Test{i}", text=remove_escapes(msg))

    perf = LexiPerformance(accuracy=float(correct_answers) / len(algo_in))
    msg = f"You got {correct_answers}/{len(algo_in)} ({perf.accuracy}) correct results!"
    r.text("ResultsInfo", text=remove_escapes(msg))
    return perf, r


def exercise1_perf_aggregator(perf_outs: List[LexiPerformance, ]) -> LexiPerformance:
    """Aggregating by average accuracy"""
    if not perf_outs:
        return LexiPerformance(accuracy=0)
    else:
        assert len(perf_outs) > 0, perf_outs
        avg_accuracy = sum([p.accuracy for p in perf_outs])/len(perf_outs)
        return LexiPerformance(accuracy=avg_accuracy)


def get_exercise1() -> Exercise:
    # test cases generation can be improved
    test_values = []
    size = [2, 6]
    expected_results = []

    for i in range(40):
        if random() > 0.1:
            values = tuple(np.round(np.random.random(size) * 10))
        else:
            v0 = np.round(np.random.random(size[1]) * 10)
            values = (v0, v0)
        test_values.append(values)
        expected_results.append(None)

    ex = Exercise[List[Tuple[np.ndarray]], List[ComparisonOutcome]](
        desc="This is exercise1 about lexicographic comparisons",
        perf_aggregator=exercise1_perf_aggregator,
        evaluation_fun=exercise1_eval,
        test_values=[test_values],
        expected_results=[expected_results],
    )
    return ex
