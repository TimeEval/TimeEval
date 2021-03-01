import unittest

import numpy as np

from timeeval.utils.metrics import roc


class TestMetrics(unittest.TestCase):

    def test_regards_nan_as_wrong(self):
        y_scores = np.array([np.nan, 0.1, 0.9])
        y_true = np.array([0, 0, 1])
        result = roc(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

        y_true = np.array([1, 0, 1])
        result = roc(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

    def test_regards_inf_as_wrong(self):
        y_scores = np.array([0.1, np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = roc(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = roc(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)
