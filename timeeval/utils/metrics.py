import argparse
from enum import Enum
from typing import Iterable, Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from prts import ts_precision, ts_recall, ts_fscore


class Metric(Enum):
    ROC_AUC = 0
    RANGE_PRECISION = 1
    RANGE_RECALL = 2
    RANGE_F1 = 3
    RANGE_PR_AUC = 4

    DEFAULT_METRICS = [ROC_AUC]

    def _validate_scores(self, scores: np.ndarray):
        if self not in [Metric.ROC_AUC, Metric.RANGE_PR_AUC]:
            if scores.dtype != np.int_:
                raise ValueError("When using Metrics other than ROC (like Precision, Recall or F1-Score), "
                                 "the scores must be integers and have the values {0, 1}."
                                 "Please consider applying a threshold to the scores!")

    def __call__(self, y_score: np.ndarray, y_true: np.ndarray, **kwargs) -> float:
        self._validate_scores(y_score)

        if self == Metric.ROC_AUC:
            return roc(y_score, y_true, **kwargs)
        elif self == Metric.RANGE_PRECISION:
            return ts_precision(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_RECALL:
            return ts_recall(y_true, y_score, **kwargs)
        elif self == Metric.RANGE_F1:
            return ts_fscore(y_true, y_score, **kwargs)
        else:  # if self == Metrics.RANGE_PR_AUC:
            return ts_precision_recall_auc(y_true, y_score)


def ts_precision_recall_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(y_score)
    thresholds.sort()
    recalls = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        y_pred = (y_score >= threshold).astype(np.int64)
        recalls[i] = ts_recall(y_true, y_pred)
        precisions[i] = ts_precision(y_true, y_pred)
    sorted_idx = recalls.argsort()
    area = auc(np.r_[0, recalls[sorted_idx]], np.r_[1, precisions[sorted_idx]])
    return area


def _substitute_nans(y_scores: Iterable[float], y_true: np.ndarray) -> np.ndarray:
    def substitute(i):
        if y_true[i] == 1:
            return 0.
        else:
            return 1.

    return np.array([
        substitute(i) if np.isinf(y) or np.isnan(y) else y
        for i, y in enumerate(y_scores)
    ])


def _metric(y_score: Iterable[float], y_true: np.ndarray, _curve_function: Callable):
    curve = _curve_function(y_true, y_score)
    area = auc(curve[0], curve[1])
    return curve, area


def roc(y_score: Iterable[float], y_true: np.ndarray, plot: bool = False) -> float:
    y_score = _substitute_nans(y_score, y_true)
    curve, area = _metric(y_score, y_true, roc_curve)
    if plot:
        fpr, tpr, _ = curve
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.title(f"ROC | area = {area:.4f}")
        plt.show()
    return area


def _create_arg_parser():
    parser = argparse.ArgumentParser(description=f"ROC Curve Plotter")

    parser.add_argument("--input-file", type=str, required=True, help="Path to input file")
    parser.add_argument("--targets-file", type=str, required=True, help="Path to targets file")
    parser.add_argument("--metric", type=Metric, default=Metric.ROC_AUC, help="Metric to plot")

    return parser


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()

    anomaly_scores = np.loadtxt(args.input_file)
    anomaly_labels = np.loadtxt(args.targets_file)

    if args.metric == Metric.ROC_AUC:
        roc(anomaly_scores, anomaly_labels, plot=True)
