import unittest

import numpy as np

from timeeval.utils.metrics import Metric, DefaultMetrics


class TestMetrics(unittest.TestCase):

    def test_regards_nan_as_wrong(self):
        y_scores = np.array([np.nan, 0.1, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, nan_is_0=False)
        self.assertEqual(0.5, result)

        y_true = np.array([1, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, nan_is_0=False)
        self.assertEqual(0.5, result)

    def test_regards_inf_as_wrong(self):
        y_scores = np.array([0.1, np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, inf_is_1=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, inf_is_1=False)
        self.assertEqual(0.5, result)

    def test_regards_neginf_as_wrong(self):
        y_scores = np.array([0.1, -np.inf, 0.9])
        y_true = np.array([0, 0, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, neginf_is_0=False)
        self.assertEqual(0.5, result)

        y_true = np.array([0, 1, 1])
        result = DefaultMetrics.ROC_AUC(y_true, y_scores, neginf_is_0=False)
        self.assertEqual(0.5, result)

    def test_range_based_f1(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_F1(y_true, y_pred)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_range_based_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_PRECISION(y_true, y_pred)
        self.assertEqual(result, 0.5)

    def test_range_based_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_RECALL(y_true, y_pred)
        self.assertEqual(result, 1)

    def test_prf1_value_error(self):
        y_pred = np.array([0, .2, .7, 0])
        y_true = np.array([0, 1, 0, 0])
        with self.assertRaises(ValueError):
            DefaultMetrics.RANGE_F1(y_true, y_pred)

    def test_range_based_pr_curve_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0.1, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9583, places=4)

    def test_range_based_pr_auc_perfect_hit(self):
        y_pred = np.array([0, 0, 0.5, 0.5, 0, 0])
        y_true = np.array([0, 0, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_pr_curve_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0, 0])
        y_true = np.array([0, 0, 1, 1, 0, 0])
        result = DefaultMetrics.PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0000, places=4)

    def test_average_precision(self):
        y_pred = np.array([0, 0.1, 1., .5, 0, 0])
        y_true = np.array([0, 1, 1, 0, 0, 0])
        result = DefaultMetrics.AVERAGE_PRECISION(y_true, y_pred)
        self.assertAlmostEqual(result, 0.8333, places=4)

    def test_fixed_range_based_pr_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0.1, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9583, places=4)
