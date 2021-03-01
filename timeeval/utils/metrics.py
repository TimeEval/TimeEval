import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import Iterable, Callable, List
import argparse
from enum import Enum


class Metrics(Enum):
    ROC = 0


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
    parser.add_argument("--metric", type=Metrics, default=Metrics.ROC, help="Metric to plot")

    return parser


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()

    anomaly_scores = np.loadtxt(args.input_file)
    anomaly_labels = np.loadtxt(args.targets_file)

    if args.metric == Metrics.ROC:
        roc(anomaly_scores, anomaly_labels, plot=True)
