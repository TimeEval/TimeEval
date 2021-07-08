import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics.CleanStartSequenceSizeHeuristic import CleanStartSequenceSizeHeuristic


class TestCleanStartSequenceSizeHeuristic(unittest.TestCase):
    def test_factor_small(self):
        heuristic = CleanStartSequenceSizeHeuristic(max_factor=0.1)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.real_test_dataset_path)
        self.assertEqual(value, int(fixtures.dataset.length * 0.1))

    def test_factor_large(self):
        heuristic = CleanStartSequenceSizeHeuristic(max_factor=1.0)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.real_test_dataset_path)
        self.assertEqual(value, 2929)
