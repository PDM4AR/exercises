from collections.abc import Iterable
from reprep import Report
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def.ex05.data import get_example_test_values
from pdm4ar.exercises_def.structures import Exercise
from pdm4ar.exercises_def.ex05.problem_def import *
from collections.abc import Iterable

from reprep import Report
from zuper_commons.text import remove_escapes

from pdm4ar.exercises_def.ex05.data import get_example_test_values
from pdm4ar.exercises_def.ex05.problem_def import *
from pdm4ar.exercises_def.structures import Exercise


def exercise_dubins_eval(
    prob: DubinsProblem,
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
                result_msg = ""
        else:
            result_msg = "Solution unavailable \n"

        msg = ""
        msg += f"Input: \t {*query,} \n"
        msg += pre_msg
        comp_out = (
            [
                *algo_out,
            ]
            if isinstance(algo_out, Iterable)
            else str(algo_out)
        )
        msg += f"Computed: \t {comp_out} \n"
        if expected is not None:
            one_of_many = False
            if isinstance(expected[i], dict):
                exp_out = ""
                exp_paths = expected[i]["opt_paths"]
                if len(exp_paths) > 1:
                    one_of_many = True
                for p in exp_paths:
                    exp_out += str(p[1])
                    if len(exp_paths) > 1:
                        exp_out += ", "
            else:
                exp_out = (
                    [
                        *expected[i],
                    ]
                    if isinstance(expected[i], Iterable)
                    else str(expected[i])
                )
            if one_of_many:
                msg += f"Expected one of: \t {exp_out} \n"
            else:
                msg += f"Expected: \t {exp_out} \n"
        msg += result_msg
        r.text(f"Query: {i + 1}", text=remove_escapes(msg))

        if prob.plot_fun is not None:
            figsize = None
            rfig = r.figure(cols=1)
            prob.plot_fun(rfig, query, algo_out, algo_out_tf, expected[i] if expected is not None else None, sucess)

    msg = f"You got {correct_answers: .3f}/{len(test_queries)} correct results!"
    perf = DubinsPerformance(
        accuracy=float(correct_answers) / len(test_queries), weight=prob.eval_weight, id_=prob.id_num
    )
    r.text("ResultsInfo", text=remove_escapes(msg))
    return perf, r


def exercise_dubins_perf_aggregator(perf_outs: List[DubinsPerformance]) -> DubinsFinalPerformance:
    accuracy_dict = {}
    accuracy_combined = sum([el.accuracy * el.weight for el in perf_outs])
    for el in perf_outs:
        if el.id_ == 1:
            accuracy_dict["accuracy_radius"] = el.accuracy
        elif el.id_ == 2:
            accuracy_dict["accuracy_turning"] = el.accuracy
        elif el.id_ == 3:
            accuracy_dict["accuracy_tangents"] = el.accuracy
        elif el.id_ == 4:
            accuracy_dict["accuracy_dubins"] = el.accuracy
        elif el.id_ == 5:
            accuracy_dict["accuracy_spline"] = el.accuracy
        elif el.id_ == 6:
            accuracy_dict["accuracy_reeds"] = el.accuracy
    return DubinsFinalPerformance(accuracy_combined=accuracy_combined, individual_accuracies=accuracy_dict)


def get_exercise5() -> Exercise:
    test_values = []
    expected_results = []
    test_configs = get_example_test_values()
    test_values = test_configs
    expected_results = None

    import pickle
    import pathlib

    path = pathlib.Path(__file__).parent.resolve()
    with open(f"{path}/expected.pickle", "rb") as f:
        expected_results = pickle.load(f)
        # print(f"test_values: {len(test_values)}, expected_results: {len(expected_results)}")  # Add this line

    return Exercise[List[DubinsProblem], Any](
        desc="This exercise is about dubins' path search",
        perf_aggregator=exercise_dubins_perf_aggregator,
        evaluation_fun=exercise_dubins_eval,
        test_values=test_values,
        expected_results=expected_results,
    )
