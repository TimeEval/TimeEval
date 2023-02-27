import unittest

from timeeval.params import FixedParameters
from timeeval.params.params import FixedParams


class TestFixedParameters(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {"a": 2, "b": False, "c": "auto"}

    def test_fixed_parameters(self):
        grid = FixedParameters(self.params)
        self.assertListEqual(list(grid), [FixedParams(self.params)])

    def test_fixed_parameters_wrong_input(self):
        with self.assertRaises(TypeError) as ex:
            FixedParameters([{"a": [1, 2]}])
        self.assertRegex(str(ex.exception), "[P|p]lease use a.*IndependentParameterGrid.")

        with self.assertRaises(TypeError) as ex:
            FixedParameters("a")
        self.assertRegex(str(ex.exception), "[P|p]arameters are not provided as a dict")
