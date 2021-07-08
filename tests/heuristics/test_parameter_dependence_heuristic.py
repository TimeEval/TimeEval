import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import ParameterDependenceHeuristic


class TestParameterDependenceHeuristic(unittest.TestCase):
    def test_both_transforms_not_supported(self):
        with self.assertRaises(ValueError):
            ParameterDependenceHeuristic(source_parameter="x",
                                         fn=lambda x: x ** 2,
                                         factor=0.5)

    def test_no_transform(self):
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(source_parameter="x")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, params=params)
        self.assertEqual(value, params["x"])

    def test_factor(self):
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(source_parameter="x", factor=2.0)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, params=params)
        self.assertEqual(value, 10)

    def test_lambda(self):
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(source_parameter="x",
                                                 fn=lambda x: f"={x}%")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, params=params)
        self.assertEqual(value, "=5%")

    def test_unset_source_parameter(self):
        params = {}
        heuristic = ParameterDependenceHeuristic(source_parameter="x")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path, params=params)
        self.assertEqual(value, None)
