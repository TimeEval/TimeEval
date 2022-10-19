import unittest
import warnings

import numpy as np

from timeeval import DefaultMetrics
from timeeval.metrics import (RangeFScore, RangePrecision, RangeRecall, F1Score, Precision, Recall, FScoreAtK,
                              PrecisionAtK, RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS)
from timeeval.metrics.thresholding import FixedValueThresholding, NoThresholding


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

    def test_range_based_f_score_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangeFScore(thresholding_strategy=FixedValueThresholding(), beta=1)(y_true, y_score)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_range_based_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_PRECISION(y_true, y_pred)
        self.assertEqual(result, 0.5)

    def test_range_based_precision_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangePrecision(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_range_based_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = DefaultMetrics.RANGE_RECALL(y_true, y_pred)
        self.assertEqual(result, 1)

    def test_range_based_recall_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = RangeRecall(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertEqual(result, 1)

    def test_rf1_value_error(self):
        y_pred = np.array([0, .2, .7, 0])
        y_true = np.array([0, 1, 0, 0])
        with self.assertRaises(ValueError):
            DefaultMetrics.RANGE_F1(y_true, y_pred)

    def test_range_based_p_range_based_r_curve_auc(self):
        y_pred = np.array([0, 0.1, 1., .5, 0.1, 0])
        y_true = np.array([0, 1, 1, 1, 0, 0])
        result = DefaultMetrics.RANGE_PR_AUC(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9583, places=4)

    def test_range_based_p_range_based_r_auc_perfect_hit(self):
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
        y_inverted = (y_true * -1 + 1).astype(np.float_)

        pr_metrics = [DefaultMetrics.PR_AUC, DefaultMetrics.RANGE_PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC]
        range_metrics = [RangeRocAUC(), RangePrAUC(), RangeRocVUS(), RangePrVUS()]
        other_metrics = [DefaultMetrics.ROC_AUC, PrecisionAtK(), FScoreAtK()]
        metrics = [*pr_metrics, *range_metrics, *other_metrics]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="Cannot compute metric for a constant value in y_score, returning 0.0!")
            for y_pred in [y_zeros, y_flat, y_ones]:
                for m in metrics:
                    self.assertAlmostEqual(m(y_true, y_pred), 0, msg=m.name)

            for m in pr_metrics:
                score = m(y_true, y_inverted)
                self.assertTrue(score <= 0.2, msg=f"{m.name}(y_true, y_inverted)={score} is not <= 0.2")
            # range metrics can deal with lag and this inverted score
            for m in other_metrics:
                score = m(y_true, y_inverted)
                self.assertAlmostEqual(score, 0, msg=m.name)

    def test_f1(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = F1Score(NoThresholding())(y_true, y_pred)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_f_score_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = F1Score(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.66666, places=4)

    def test_precision(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Precision(NoThresholding())(y_true, y_pred)
        self.assertEqual(result, 0.5)

    def test_precision_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = Precision(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_recall(self):
        y_pred = np.array([0, 1, 1, 0])
        y_true = np.array([0, 1, 0, 0])
        result = Recall(NoThresholding())(y_true, y_pred)
        self.assertEqual(result, 1)

    def test_recall_thresholding(self):
        y_score = np.array([0.1, 0.9, 0.8, 0.2])
        y_true = np.array([0, 1, 0, 0])
        result = Recall(thresholding_strategy=FixedValueThresholding())(y_true, y_score)
        self.assertEqual(result, 1)


class TestVUSMetrics(unittest.TestCase):
    def setUp(self) -> None:
        y_true = np.zeros(200)
        y_true[10:20] = 1
        y_true[28:33] = 1
        y_true[110:120] = 1
        y_score = np.random.default_rng(41).random(200) * 0.5
        y_score[16:22] = 1
        y_score[33:38] = 1
        y_score[160:170] = 1
        self.y_true = y_true
        self.y_score = y_score
        self.expected_range_pr_auc = 0.3737854660
        self.expected_range_roc_auc = 0.7108527197
        self.expected_range_pr_volume = 0.7493254559  # max_buffer_size = 200
        self.expected_range_roc_volume = 0.8763382130  # max_buffer_size = 200

    def test_range_pr_auc_compat(self):
        result = RangePrAUC(compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_pr_auc, places=10)

    def test_range_roc_auc_compat(self):
        result = RangeRocAUC(compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_roc_auc, places=10)

    def test_edge_case_existence_reward_compat(self):
        result = RangePrAUC(compatibility_mode=True, buffer_size=4)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.2506464391, places=10)
        result = RangeRocAUC(compatibility_mode=True, buffer_size=4)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.6143220816, places=10)

    def test_range_pr_volume_compat(self):
        result = RangePrVUS(max_buffer_size=200, compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_pr_volume, places=10)

    def test_range_roc_volume_compat(self):
        result = RangeRocVUS(max_buffer_size=200, compatibility_mode=True)(self.y_true, self.y_score)
        self.assertAlmostEqual(result, self.expected_range_roc_volume, places=10)

    def test_range_pr_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangePrAUC()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9636, places=4)

    def test_range_roc_auc(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeRocAUC()(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9653, places=4)

    def test_range_pr_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangePrVUS(max_buffer_size=200)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9937, places=4)

    def test_range_roc_volume(self):
        y_pred = np.array([0.05, 0.2, 1., 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.07])
        y_true = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        result = RangeRocVUS(max_buffer_size=200)(y_true, y_pred)
        self.assertAlmostEqual(result, 0.9904, places=4)
