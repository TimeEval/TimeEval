import unittest
import warnings

import numpy as np

from timeeval import DefaultMetrics
from timeeval.metrics.other_metrics import FScoreAtK, PrecisionAtK


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
        self.assertAlmostEqual(result, 1.0000, places=4)

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
        result = DefaultMetrics.FIXED_RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9792, places=4)

    def test_range_based_pr_auc_discrete(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.000, places=4)

    def test_fixed_range_based_pr_auc_discrete(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = DefaultMetrics.FIXED_RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 1.000, places=4)

    def test_precision_at_k(self):
        y_pred = np.array([0, 0.1, 1., .6, 0.1, 0, 0.4, 0.5])
        y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        result = PrecisionAtK()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.5000, places=4)
        result = PrecisionAtK(k=1)(y_true, y_pred)
        self.assertAlmostEqual(result, 1.0000, places=4)

    def test_fscore_at_k(self):
        y_pred = np.array([0.4, 0.1, 1., .5, 0.1, 0, 0.4, 0.5])
        y_true = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        result = FScoreAtK()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.500, places=4)
        result = FScoreAtK(k=3)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.800, places=4)

    def test_edge_cases(self):
        y_true = np.zeros(10, dtype=np.int_)
        y_true[2:4] = 1
        y_true[6:8] = 1
        y_zeros = np.zeros_like(y_true, dtype=np.float_)
        y_flat = np.full_like(y_true, fill_value=0.5, dtype=np.float_)
        y_ones = np.ones_like(y_true, dtype=np.float_)
        y_inverted = (y_true*-1+1).astype(np.float_)

        pr_metrics = [DefaultMetrics.PR_AUC, DefaultMetrics.RANGE_PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC]
        other_metrics = [DefaultMetrics.ROC_AUC, PrecisionAtK(), FScoreAtK()]
        metrics = [*pr_metrics, *other_metrics]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="Cannot compute metric for a constant value in y_score, returning 0.0!")
            for y_pred in [y_zeros, y_flat, y_ones]:
                for m in metrics:
                    self.assertAlmostEqual(m(y_true, y_pred), 0, msg=m.name)

            for m in pr_metrics:
                score = m(y_true, y_inverted)
                self.assertTrue(score <= 0.2, msg=f"{m.name}(y_true, y_inverted)={score} is not <= 0.2")
            for m in other_metrics:
                score = m(y_true, y_inverted)
                self.assertAlmostEqual(score, 0, msg=m.name)
