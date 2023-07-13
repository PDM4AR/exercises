import argparse
import os
import logging

import contracts
from zuper_commons.types import ZValueError
from zuper_commons.text import remove_escapes
from zuper_typing import debug_print

from pdm4ar.available_exercises import available_exercises
from pdm4ar.exercises_def.structures import Exercise, ExerciseEvaluator, out_dir
from pdm4ar.exercises_def import logger


def find_exercise(exercise: str, evaluation_mode=False) -> Exercise:
    if evaluation_mode:
        from pdm4ar_sol import available_evaluations
        if exercise not in available_evaluations:
            raise ZValueError(f"Cannot find {exercise!r}", available=set(available_evaluations))
        return available_evaluations[exercise]()
    else:
        if exercise not in available_exercises:
            raise ZValueError(f"Cannot find {exercise!r}", available=set(available_exercises))
        return available_exercises[exercise]()

def run_exercise(exercise: str, evaluation_mode=False):
    ex: Exercise = find_exercise(exercise, evaluation_mode)
    evaluator = ExerciseEvaluator(exercise=ex)

    out_main = out_dir(exercise)
    logging.getLogger("reprep").setLevel(logging.WARNING)   # suppress annoying messages from reprep

    total = len(ex.test_values)
    # for i, alg_in in enumerate(tqdm(ex.test_values)):

    perf, report = evaluator.evaluate()

    report_file = os.path.join(out_main, "index.html")
    report.to_html(report_file)
    result_str = remove_escapes(debug_print(perf))
    logger.info(f"\n<<<<<\n{result_str}\n>>>>>")


def _setup_args():
    parser = argparse.ArgumentParser(description="PDM4AR exercise")
    parser.add_argument("-e", "--exercise", help="name of the exercise to run", type=str)
    parser.add_argument("-g", "--evaluate", help="(not for student) run evaluation test cases", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    contracts.disable_all()
    args = _setup_args()
    run_exercise(exercise=args.exercise, evaluation_mode=args.evaluate)
