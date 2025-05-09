import unittest
import warnings

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import ParameterDependenceHeuristic
from timeeval.heuristics.base import HeuristicFallbackWarning


class TestParameterDependenceHeuristic(unittest.TestCase):
    def setUp(self) -> None:
        warnings.simplefilter("ignore", category=HeuristicFallbackWarning)

    def tearDown(self) -> None:
        warnings.simplefilter("default", category=HeuristicFallbackWarning)

    def test_both_transforms_not_supported(self) -> None:
        with self.assertRaises(ValueError):
            ParameterDependenceHeuristic(
                source_parameter="x", fn=lambda x: x**2, factor=0.5
            )

    def test_no_transform(self) -> None:
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(source_parameter="x")
        value = heuristic(
            fixtures.algorithm,
            fixtures.dataset,
            fixtures.dummy_dataset_path,
            params=params,
        )
        self.assertEqual(value, params["x"])

    def test_factor(self) -> None:
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(source_parameter="x", factor=2.0)
        value = heuristic(
            fixtures.algorithm,
            fixtures.dataset,
            fixtures.dummy_dataset_path,
            params=params,
        )
        self.assertEqual(value, 10)

    def test_lambda(self) -> None:
        params = {"x": 5}
        heuristic = ParameterDependenceHeuristic(
            source_parameter="x", fn=lambda x: f"={x}%"
        )
        value = heuristic(
            fixtures.algorithm,
            fixtures.dataset,
            fixtures.dummy_dataset_path,
            params=params,
        )
        self.assertEqual(value, "=5%")

    def test_unset_source_parameter(self) -> None:
        params = {}
        heuristic = ParameterDependenceHeuristic(source_parameter="x")
        value = heuristic(
            fixtures.algorithm,
            fixtures.dataset,
            fixtures.dummy_dataset_path,
            params=params,
        )
        self.assertEqual(value, None)
