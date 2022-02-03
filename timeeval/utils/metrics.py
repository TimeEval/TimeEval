import warnings
from enum import Enum
from typing import Iterable, Callable, List, Tuple

import numpy as np
from prts import ts_precision, ts_recall, ts_fscore
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length


class Metric(Enum):
    ROC_AUC = 0
    PR_AUC = 1
    RANGE_PR_AUC = 2
    RANGE_PRECISION = 3
    RANGE_RECALL = 4
    RANGE_F1 = 5
    AVERAGE_PRECISION = 6

    @staticmethod
    def default() -> 'Metric':
        return Metric.ROC_AUC

    @staticmethod
    def default_list() -> List['Metric']:
        return [Metric.ROC_AUC]

    def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
                         inf_is_1: bool = True,
                         neginf_is_0: bool = True,
                         nan_is_0: bool = True,
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                          "y_true should be an integer array and y_score a float array!")
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if self not in [Metric.ROC_AUC, Metric.PR_AUC, Metric.RANGE_PR_AUC, Metric.AVERAGE_PRECISION]:
            if y_score.dtype != np.int_:
                raise ValueError("When using Metrics other than AUC-metric (like Precision, Recall or F1-Score), "
                                 "the scores must be integers and have the values {0, 1}. Please consider applying "
                                 "a threshold to the scores!")
        else:
            if y_score.dtype != np.float_:
                raise ValueError("When using AUC-based metrics, the scores must be floats!")

        # substitute NaNs and Infs
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        penalize_mask = np.full_like(y_score, dtype=bool, fill_value=False)
        if inf_is_1:
            y_score[inf_mask] = 1
        else:
            penalize_mask = penalize_mask | inf_mask
        if neginf_is_0:
            y_score[neginf_mask] = 0
        else:
            penalize_mask = penalize_mask | neginf_mask
        if nan_is_0:
            y_score[nan_mask] = 0.
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)

        if self == Metric.ROC_AUC:
            return auc_roc(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_PRECISION:
            return ts_precision(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_RECALL:
            return ts_recall(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_F1:
            return ts_fscore(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_PR_AUC:
            return auc_range_precision_recall_curve(y_true, y_score, **kwargs)
        elif self == Metric.PR_AUC:
            return auc_precision_recall_curve(y_true, y_score, **kwargs)
        else:  # if self == Metric.AVERAGE_PRECISION:
            kwargs.pop("plot", None)
            kwargs.pop("plot_store", None)
            return average_precision_score(y_true, y_score, pos_label=1, **kwargs)


def _metric(y_true: np.ndarray, y_score: Iterable[float], _curve_function: Callable, plot: bool = False, **kwargs) -> float:
    x, y, thresholds = _curve_function(y_true, y_score)
    if "precision_recall" in _curve_function.__name__:
        # swap x and y
        x, y = y, x
    area = auc(x, y)
    if plot:
        store = False
        if "plot_store" in kwargs:
            store = kwargs["plot_store"]

        import matplotlib.pyplot as plt

        name = _curve_function.__name__
        plt.plot(x, y, label=name, drawstyle="steps-post")
        # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.title(f"{name} | area = {area:.4f}")
        if store:
            plt.savefig(f"fig-{name}.pdf")
        plt.show()
    return area


def range_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, max_samples=50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.unique(y_score)
    thresholds.sort()

    # sample thresholds
    n_thresholds = thresholds.shape[0]
    if n_thresholds > max_samples:
        every_nth = n_thresholds // (max_samples - 1)
        sampled_thresholds = thresholds[::every_nth]
        if thresholds[-1] == sampled_thresholds[-1]:
            thresholds = sampled_thresholds
        else:
            thresholds = np.r_[sampled_thresholds, thresholds[-1]]

    recalls = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(np.int64)
        recalls[i] = ts_recall(y_true, y_pred)
        precisions[i] = ts_precision(y_true, y_pred)
    sorted_idx = recalls.argsort()[::-1]
    return np.r_[precisions[sorted_idx], 1], np.r_[recalls[sorted_idx], 0], thresholds


def auc_roc(y_true: np.ndarray, y_score: Iterable[float], **kwargs) -> float:
    return _metric(y_true, y_score, roc_curve, **kwargs)


def auc_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    return _metric(y_true, y_score, precision_recall_curve, **kwargs)


def auc_range_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    return _metric(y_true, y_score, range_precision_recall_curve, **kwargs)
