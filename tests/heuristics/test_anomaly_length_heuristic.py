import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import AnomalyLengthHeuristic


class TestAnomalyLengthHeuristic(unittest.TestCase):
    def test_wrong_agg_type(self):
        with self.assertRaises(ValueError):
            AnomalyLengthHeuristic(agg_type="wrong")

    def test_min(self):
        heuristic = AnomalyLengthHeuristic(agg_type="min")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, fixtures.dataset.min_anomaly_length)

    def test_median(self):
        heuristic = AnomalyLengthHeuristic(agg_type="median")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, fixtures.dataset.median_anomaly_length)

    def test_max(self):
        heuristic = AnomalyLengthHeuristic(agg_type="max")
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, fixtures.dataset.max_anomaly_length)
