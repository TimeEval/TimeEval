import unittest

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics.ContaminationHeuristic import ContaminationHeuristic


class TestContaminationHeuristic(unittest.TestCase):
    def test_heuristic(self):
        heuristic = ContaminationHeuristic()
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.real_test_dataset_path)
        self.assertEqual(value, 1.0 / 3600)
