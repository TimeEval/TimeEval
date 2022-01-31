import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import DatasetIdHeuristic


class TestDatasetIdHeuristic(unittest.TestCase):
    def test_heuristic(self):
        heuristic = DatasetIdHeuristic()
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.real_test_dataset_path)
        self.assertEqual(value, fixtures.dataset.datasetId)
