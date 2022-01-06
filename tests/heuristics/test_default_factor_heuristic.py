import unittest
from copy import deepcopy

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import DefaultFactorHeuristic


class TestDefaultFactorHeuristic(unittest.TestCase):

    def setUp(self) -> None:
        self.algo = deepcopy(fixtures.algorithm)
        self.alpha_default = 0.1
        self.algo.param_schema = {
            "alpha": {
                "defaultValue": self.alpha_default,
                "description": "Test parameter",
                "name": "alpha",
                "type": "float"
            }}

    def test_no_factor(self):
        heuristic = DefaultFactorHeuristic()
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertEqual(value, self.alpha_default)

    def test_factor(self):
        heuristic = DefaultFactorHeuristic(factor=2.0)
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertEqual(value, 2*self.alpha_default)

    def test_zero_fb(self):
        self.algo.param_schema["alpha"]["defaultValue"] = 0

        heuristic = DefaultFactorHeuristic(factor=2.0, zero_fb=10)
        value = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        heuristic = DefaultFactorHeuristic(factor=2.0)
        value2 = heuristic(self.algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.algo.param_schema["alpha"]["defaultValue"] = self.alpha_default

        self.assertEqual(value, 20)
        self.assertEqual(value2, 2)

    def test_default_value_not_available(self):
        heuristic = DefaultFactorHeuristic(factor=2.0)
        with self.assertRaises(ValueError) as e:
            heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertIn("Could not find the default value", str(e.exception))

    def test_returns_expected_data_type_int(self):
        algo = deepcopy(self.algo)
        algo.param_schema["alpha"]["defaultValue"] = 1
        heuristic = DefaultFactorHeuristic(factor=2.0)
        value = heuristic(algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertEqual(type(value), int)

    def test_returns_expected_data_type_float(self):
        algo = deepcopy(self.algo)
        algo.param_schema["alpha"]["defaultValue"] = 1.0
        heuristic = DefaultFactorHeuristic(factor=2.0)
        value = heuristic(algo, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertEqual(type(value), float)
