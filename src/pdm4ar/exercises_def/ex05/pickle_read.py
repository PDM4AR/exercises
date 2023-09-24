import pickle
import os

expected_file = f = open(os.path.join(os.path.dirname(__file__), 'expected.pickle'), "rb")
expected = pickle.load(expected_file)
expected_file.close()
print(expected[2])