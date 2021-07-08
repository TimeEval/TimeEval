import unittest
from copy import deepcopy

import numpy as np

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import PeriodSizeHeuristic


class TestPeriodSizeHeuristic(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_none = deepcopy(fixtures.dataset)
        self.dataset_none.period_size = None
        # Pandas sometimes returns NaNs instead of None if it's not set:
        self.dataset_nan = deepcopy(fixtures.dataset)
        self.dataset_nan.period_size = np.nan  # type: ignore

    def test_factor(self):
        heuristic = PeriodSizeHeuristic(factor=1.5)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, int(fixtures.dataset.period_size * 1.5))

    def test_anomaly_length_fallback(self):
        heuristic = PeriodSizeHeuristic(fb_anomaly_length_agg_type="median")
        value = heuristic(fixtures.algorithm, self.dataset_none, fixtures.dummy_dataset_path)
        self.assertEqual(value, fixtures.dataset.median_anomaly_length)

    def test_wrong_agg_type_fallback(self):
        heuristic = PeriodSizeHeuristic(fb_anomaly_length_agg_type="wrong")
        value = heuristic(fixtures.algorithm, self.dataset_none, fixtures.dummy_dataset_path)
        self.assertEqual(value, 1)

    def test_value_fallback(self):
        heuristic = PeriodSizeHeuristic(fb_value=213)
        value = heuristic(fixtures.algorithm, self.dataset_none, fixtures.dummy_dataset_path)
        self.assertEqual(value, 213)

    def test_value_fallback_on_nan(self):
        heuristic = PeriodSizeHeuristic(fb_value=213)
        value = heuristic(fixtures.algorithm, self.dataset_nan, fixtures.dummy_dataset_path)
        self.assertEqual(value, 213)
