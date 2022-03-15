import os
import unittest

import numpy as np
import pytest

from tests.fixtures.algorithms import DeviatingFromMedian
from timeeval.adapters import MultivarAdapter
from timeeval.adapters.multivar import AggregationMethod


class TestMultivarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(100, 3)
        tmp = np.abs(self.X - np.median(self.X, axis=0))
        self.y = tmp / tmp.max(axis=0)
        self.y_median = np.median(self.y, axis=1)
        self.y_mean = np.mean(self.y, axis=1)
        self.y_max = np.max(self.y, axis=1)

    def test_multivar_deviating_from_median_mean(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MEAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_mean, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_median(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MEDIAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_median, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_max(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MAX)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_max, score)
        self.assertEqual(len(self.X), len(score))

    @pytest.mark.skipif(condition=os.getenv("CI", "false") == "true", reason="CI never finishes on sopedu")
    def test_multivar_deviating_from_median_parallel(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MEAN, n_jobs=2)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_mean, score)
        self.assertEqual(len(self.X), len(score))
