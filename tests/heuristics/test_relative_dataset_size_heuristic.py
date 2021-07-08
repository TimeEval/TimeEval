import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import RelativeDatasetSizeHeuristic


class TestRelativeDatasetSizeHeuristic(unittest.TestCase):
    def test_factor_small(self):
        heuristic = RelativeDatasetSizeHeuristic(factor=0.1)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, int(fixtures.dataset.length * 0.1))

    def test_factor_large(self):
        heuristic = RelativeDatasetSizeHeuristic(factor=0.9)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, int(fixtures.dataset.length * 0.9))
