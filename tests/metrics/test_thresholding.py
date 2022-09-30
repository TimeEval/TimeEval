import unittest

import numpy as np

from timeeval.metrics.thresholding import FixedValueThresholding, PercentileThresholding, TopKPointsThresholding, \
    TopKRangesThresholding, SigmaThresholding, NoThresholding


class TestThresholding(unittest.TestCase):

    def setUp(self) -> None:
        self.y_scores = np.array([np.nan, 0.1, 0.9, 0.5, 0.6, 0.55, 0.2, 0.1, np.nan])
        self.y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

    def _test_strategy(self, strategy, expected_threshold, expected_result):
        expected_result = np.array(expected_result)
        threshold = strategy.find_threshold(self.y_true, self.y_scores)
        self.assertAlmostEqual(expected_threshold, threshold, places=2)
        result = strategy.fit_transform(self.y_true, self.y_scores)
        np.testing.assert_array_equal(expected_result, result)

    def test_no_thresholding(self):
        strategy = NoThresholding()
        self.assertIsNone(strategy.find_threshold(self.y_true, self.y_true))
        result = strategy.fit_transform(self.y_true, self.y_true)
        np.testing.assert_array_equal(self.y_true, result)

    def test_no_thresholding_error(self):
        strategy = NoThresholding()
        with self.assertRaises(ValueError) as ex:
            strategy.fit_transform(self.y_true, self.y_scores)
        self.assertIn("Continuous anomaly scorings are not supported", str(ex.exception))

    def test_fixed_value_thresholding(self):
        strategy = FixedValueThresholding(threshold=0.7)
        self._test_strategy(strategy, 0.7, [0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_percentile_thresholding(self):
        strategy = PercentileThresholding(percentile=90)
        self._test_strategy(strategy, 0.72, [0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_top_k_points_thresholding(self):
        strategy = TopKPointsThresholding(k=2)
        self._test_strategy(strategy, 0.58, [0, 0, 1, 0, 1, 0, 0, 0, 0])

    def test_top_k_ranges_thresholding(self):
        strategy = TopKRangesThresholding(k=2)
        self._test_strategy(strategy, 0.60, [0, 0, 1, 0, 1, 0, 0, 0, 0])

    def test_sigma_thresholding(self):
        strategy = SigmaThresholding(factor=1)
        self._test_strategy(strategy, 0.70, [0, 0, 1, 0, 0, 0, 0, 0, 0])
