import unittest

import numpy as np
import pytest

from timeeval.metrics.thresholding import (
    FixedValueThresholding,
    NoThresholding,
    PercentileThresholding,
    PyThreshThresholding,
    SigmaThresholding,
    TopKPointsThresholding,
    TopKRangesThresholding,
)

try:
    import pythresh  # noqa: F401

    _skip_pythresh_test = False
except ImportError:
    _skip_pythresh_test = True


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
        self.assertEqual(strategy.find_threshold(self.y_true, self.y_true), 0.5)

        # allow ints
        result = strategy.fit_transform(self.y_true, self.y_true)
        np.testing.assert_array_equal(self.y_true, result)

        # allow bools
        result = strategy.fit_transform(self.y_true, self.y_true.astype(np.bool_))
        np.testing.assert_array_equal(self.y_true, result)

    def test_no_thresholding_error(self):
        strategy = NoThresholding()
        with self.assertRaises(ValueError) as ex:
            strategy.fit_transform(self.y_true, self.y_scores)
        self.assertIn(
            "Continuous anomaly scorings are not supported", str(ex.exception)
        )

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

    @pytest.mark.skipif(
        _skip_pythresh_test is True, reason="PyThresh is not installed!"
    )
    def test_pythresh_thresholding(self):
        import pythresh.version
        from pythresh.thresholds.regr import REGR

        with self.assertWarnsRegex(DeprecationWarning, "parameter is deprecated"):
            strategy = PyThreshThresholding(
                pythresh_thresholder=REGR(method="theil"), random_state=42
            )

        pythresh_version = list(map(int, pythresh.version.__version__.split(".")))
        if pythresh_version >= [0, 2, 8]:
            exp_threshold = 0.72
            exp_res = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        else:
            exp_threshold = 0.44
            exp_res = [0, 0, 1, 1, 1, 1, 0, 0, 0]
        self._test_strategy(strategy, exp_threshold, exp_res)

    @pytest.mark.skipif(
        _skip_pythresh_test is True, reason="PyThresh is not installed!"
    )
    def test_pythresh_thresholding_new(self):
        import pythresh.version
        from pythresh.thresholds.regr import REGR

        pythresh_version = list(map(int, pythresh.version.__version__.split(".")))
        if pythresh_version >= [0, 2, 8]:
            strategy = PyThreshThresholding(
                pythresh_thresholder=REGR(method="theil", random_state=42)
            )
            self._test_strategy(strategy, 0.72, [0, 0, 1, 0, 0, 0, 0, 0, 0])
