from numpy import empty
import math
from dataclasses import dataclass
from abc import ABC
from typing import Callable, Tuple, Any
from collections.abc import Iterable
from reprep import Report, MIME_PDF
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def.ex05.comparison import *
#from pdm4ar.exercises_def.ex05.data import *
from pdm4ar.exercises_def.structures import Exercise
from pdm4ar.exercises_def.ex05.problem_def import *

def exercise_dubins_eval(prob: DubinsProblem,
                         expected: List[DubinsProblem],
                         ) -> Tuple[DubinsPerformance, Report]:
    
    test_queries = prob.queries
    correct_answers = 0
   
    r = Report(prob.id_str)
    for i, query in enumerate(test_queries):
        sucess = False
        algo_out = prob.algo_fun(*query)

        if prob.pre_tf_fun is not None:
            pre_success, algo_out_tf, pre_msg = prob.pre_tf_fun(algo_out)
        else:
            pre_success = True
            algo_out_tf = None
            pre_msg = ""

        if expected is not None:
            if pre_success:
                is_correct, result_msg = prob.eval_fun(algo_out, algo_out_tf, expected[i]) 
                sucess = bool(int(is_correct))
                correct_answers += is_correct
        else:
            result_msg = "Solution unavailable \n"

        if prob.plot_fun is not None:
            figsize = None
            rfig = r.figure(cols=1)
            prob.plot_fun(rfig, query, algo_out, algo_out_tf, expected[i] if expected is not None else None, sucess)
        
        msg = ""
        msg += f"Input: \t {*query,} \n"
        msg += pre_msg
        comp_out = [*algo_out,] if isinstance(algo_out, Iterable) else str(algo_out)
        msg += f"Computed: \t {comp_out} \n"
        if expected is not None:
            if isinstance(expected[i], dict):
                exp_out = [p[1] for p in expected[i]["opt_paths"]]
            else:
                exp_out = (*expected[i],) if isinstance(expected[i], Iterable) else str(expected[i])
            msg += f"Expected: \t {exp_out} \n"
        msg += result_msg
        r.text(f"Query: {i + 1}", text=remove_escapes(msg))


    msg = f"You got {correct_answers: .3f}/{len(test_queries)} correct results!"
    perf = DubinsPerformance(accuracy=float(correct_answers) / len(test_queries), weight=prob.eval_weight)
    r.text("ResultsInfo", text=remove_escapes(msg))
    return perf, r


def exercise_dubins_perf_aggregator(perf_outs: List[DubinsPerformance]) -> DubinsPerformance:
    if len(perf_outs):
        return DubinsPerformance(sum([el.accuracy*el.weight for el in perf_outs]) / len(perf_outs), weight = 1)
    else:
        return DubinsPerformance(0, weight=1)


def get_exercise5()-> Exercise:
    test_values = []
    expected_results = []
    test_configs = get_test_values()
    test_values = test_configs
    expected_results = None

    return Exercise[List[DubinsProblem], Any](
        desc="This exercise is about dubins' path search",
        perf_aggregator=exercise_dubins_perf_aggregator,
        evaluation_fun=exercise_dubins_eval,
        test_values=test_values,
        expected_results=expected_results,
    )
