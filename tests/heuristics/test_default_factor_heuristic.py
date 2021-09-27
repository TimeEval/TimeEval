import unittest
from copy import deepcopy

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import DefaultFactorHeuristic


class TestDefaultFactorHeuristic(unittest.TestCase):

    def setUp(self) -> None:
        self.algo = deepcopy(fixtures.algorithm)
        self.alpha_default = 0.1
        self.algo.params = {
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

    def test_default_value_not_available(self):
        heuristic = DefaultFactorHeuristic(factor=2.0)
        with self.assertRaises(ValueError) as e:
            heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, param_name="alpha")
        self.assertIn("Could not find the default value", str(e.exception))
