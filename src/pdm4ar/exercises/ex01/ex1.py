from typing import NewType, Tuple

ComparisonOutcome = NewType("ComparisonOutcome", str)
""" The type of comparison outcomes """
FIRST_PREFERRED = ComparisonOutcome("first_preferred")
""" We prefer the first option. """
SECOND_PREFERRED = ComparisonOutcome("second_preferred")
""" We prefer the second option. """
INDIFFERENT = ComparisonOutcome("indifferent")
""" We are indifferent among the two options. """


def compare_lexicographic(a: Tuple[float], b: Tuple[float]) -> ComparisonOutcome:
    """
    Implement here your solution.
    The two tuples represent two vectors of outcomes (e.g. different cost function realizations) for two different decisions.
    Which one is preferred?

    Note that the terms are sorted lexicographically by importance.
    For example, the term in position 1 is less important than the one in position 0,
    but more important than the one in position 2
    """
    # todo

    return INDIFFERENT
