import unittest

import numpy as np

from timeeval.utils.metrics import Metric


class TestMetrics(unittest.TestCase):

    def test_regards_nan_as_wrong(self):
        y_scores = np.array([np.nan, 0.1, 0.9])
        y_true = np.array([0, 0, 1])
        result = Metric.ROC_AUC(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

        y_true = np.array([1, 0, 1])
        result = Metric.ROC_AUC(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

    def test_regards_inf_as_wrong(self):
        y_scores = np.array([0.1, np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = Metric.ROC_AUC(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = Metric.ROC_AUC(y_scores, y_true, plot=False)
        self.assertEqual(0.5, result)

    def test_range_based_f1(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Metric.RANGE_F1(y_pred, y_true)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_range_based_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Metric.RANGE_PRECISION(y_pred, y_true)
        self.assertEqual(result, 0.5)

    def test_range_based_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Metric.RANGE_RECALL(y_pred, y_true)
        self.assertEqual(result, 1)

    def test_prf1_value_error(self):
        y_pred = np.array([0, .2, .7, 0])
        y_true = np.array([0, 1, 0, 0])
        with self.assertRaises(ValueError):
            Metric.RANGE_F1(y_pred, y_true)

    def test_range_based_pr_curve(self):
        y_pred = np.array([0, 0.1, 1.,.5, 0, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = Metric.RANGE_PR_AUC(y_pred, y_true)
        self.assertAlmostEqual(result, 0.91666, places=4)