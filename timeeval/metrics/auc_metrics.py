from abc import ABC
from typing import Iterable, Callable

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve

from .metric import Metric


class AucMetric(Metric, ABC):
    """Base class for area-under-curve-based metrics.

    All AUC-Metrics support continuous scorings, calculate the area under a curve function, and allow plotting this
    curve function. See the subclasses' documentation for a detailed explanation of the corresponding curve and metric.
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        self._plot = plot
        self._plot_store = plot_store

    def _auc(self, y_true: np.ndarray, y_score: Iterable[float], _curve_function: Callable) -> float:
        x, y, thresholds = _curve_function(y_true, y_score)
        if "precision_recall" in _curve_function.__name__:
            # swap x and y
            x, y = y, x
        area: float = auc(x, y)
        if self._plot:
            import matplotlib.pyplot as plt

            name = _curve_function.__name__
            plt.plot(x, y, label=name, drawstyle="steps-post")
            # plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
            plt.title(f"{name} | area = {area:.4f}")
            if self._plot_store:
                plt.savefig(f"fig-{name}.pdf")
            plt.show()
        return area

    def supports_continuous_scorings(self) -> bool:
        return True


class RocAUC(AucMetric):
    """Computes the area under the receiver operating characteristic curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".

    See Also
    --------
    `https://en.wikipedia.org/wiki/Receiver_operating_characteristic <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_ : Explanation of the ROC-curve.
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, roc_curve)

    @property
    def name(self) -> str:
        return "ROC_AUC"


class PrAUC(AucMetric):
    """Computes the area under the precision recall curve.

    Parameters
    ----------
    plot : bool
        Set this parameter to ``True`` to plot the curve.
    plot_store : bool
        If this parameter is ``True`` the curve plot will be saved in the current working directory under the name
        template "fig-{metric-name}.pdf".
    """
    def __init__(self, plot: bool = False, plot_store: bool = False) -> None:
        super().__init__(plot, plot_store)

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, precision_recall_curve)

    @property
    def name(self) -> str:
        return "PR_AUC"
