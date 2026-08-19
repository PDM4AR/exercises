def get_exercise6():
    """Load the Exercise 06 definition lazily to avoid student-module import cycles."""
    from pdm4ar.exercises_def.ex06.ex06 import get_exercise6 as build_exercise6

    return build_exercise6()
