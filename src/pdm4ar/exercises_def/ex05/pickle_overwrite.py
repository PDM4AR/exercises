import pickle
import os
from dg_commons import SE2Transform
from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.problem_def import *


expected_file = open(os.path.join(os.path.dirname(__file__), 'expected.pickle'), "rb")
expected = pickle.load(expected_file)
expected_file.close()

expected[2] = [[Line(SE2Transform([2.0, 0], math.pi/2), SE2Transform([2.0, 3.0], math.pi/2), Gear.FORWARD)],
                           [Line(SE2Transform([-1.2, 1.6], 0.6435011087932844), SE2Transform([2.8, 4.6], 0.6435011087932844), Gear.FORWARD)],
                            []]

overwrite_file = open(os.path.join(os.path.dirname(__file__), 'expected.pickle'), "wb")
pickle.dump(expected, overwrite_file)
overwrite_file.close()