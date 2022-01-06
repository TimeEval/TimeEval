import unittest
from copy import deepcopy

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import DefaultExponentialFactorHeuristic


class TestDefaultExponentialFactorHeuristic(unittest.TestCase):

    def setUp(self) -> None:
        self.algo = deepcopy(fixtures.algorithm)
        self.beta_default = 10
        self.algo.param_schema = {
            "beta": {
                "defaultValue": self.beta_default,
                "description": "Test parameter",
                "name": "beta",
                "type": "float"
            }}

    def test_no_factor(self):
        heuristic = DefaultExponentialFactorHeuristic()
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.assertEqual(value, self.beta_default)

    def test_factor(self):
        heuristic = DefaultExponentialFactorHeuristic(exponent=2)
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.assertEqual(value, self.beta_default * 100)

    def test_zero_fb(self):
        self.algo.param_schema["beta"]["defaultValue"] = 0

        heuristic = DefaultExponentialFactorHeuristic(exponent=2, zero_fb=10)
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        heuristic = DefaultExponentialFactorHeuristic(exponent=2)
        value2 = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.algo.param_schema["beta"]["defaultValue"] = self.beta_default

        self.assertEqual(value, 1e3)
        self.assertEqual(value2, 1e2)

    def test_default_value_not_available(self):
        heuristic = DefaultExponentialFactorHeuristic(exponent=2)
        with self.assertRaises(ValueError) as e:
            heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.assertIn("Could not find the default value", str(e.exception))

    def test_returns_expected_data_type_int(self):
        algo = deepcopy(self.algo)
        algo.param_schema["beta"]["defaultValue"] = 1
        heuristic = DefaultExponentialFactorHeuristic(exponent=2)
        value = heuristic(algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.assertEqual(type(value), int)

    def test_returns_expected_data_type_float(self):
        algo = deepcopy(self.algo)
        algo.param_schema["beta"]["defaultValue"] = 1.0
        heuristic = DefaultExponentialFactorHeuristic(exponent=2)
        value = heuristic(algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="beta")
        self.assertEqual(type(value), float)
