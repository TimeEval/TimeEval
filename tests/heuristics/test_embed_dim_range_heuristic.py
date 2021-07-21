import unittest
from copy import deepcopy

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import EmbedDimRangeHeuristic


class TestEmbedDimRangeHeuristic(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_none = deepcopy(fixtures.dataset)
        self.dataset_none.period_size = None

    def test_default(self):
        heuristic = EmbedDimRangeHeuristic()
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, [
            int(fixtures.dataset.period_size * 0.5),
            fixtures.dataset.period_size,
            int(fixtures.dataset.period_size * 1.5)
        ])

    def test_fb_value(self):
        heuristic = EmbedDimRangeHeuristic(base_fb_value=10)
        value = heuristic(fixtures.algorithm, self.dataset_none, fixtures.dummy_dataset_path)
        self.assertEqual(value, [5, 10, 15])

    def test_base_factor(self):
        heuristic = EmbedDimRangeHeuristic(base_factor=2)
        value = heuristic(fixtures.algorithm, fixtures.dataset, fixtures.dummy_dataset_path)
        self.assertEqual(value, [
            fixtures.dataset.period_size,
            int(fixtures.dataset.period_size * 2),
            int(fixtures.dataset.period_size * 3)
        ])

    def test_range_factors(self):
        heuristic = EmbedDimRangeHeuristic(dim_factors=[1,1.,2.5])
        value = heuristic(fixtures.algorithm, self.dataset_none, fixtures.dummy_dataset_path)
        self.assertEqual(value, [50, 50, 125])
