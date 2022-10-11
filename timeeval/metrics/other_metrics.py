from typing import Optional

import numpy as np
from prts import ts_fscore, ts_precision
from sklearn.metrics import average_precision_score

from .metric import Metric
from .thresholding import TopKRangesThresholding


class AveragePrecision(Metric):
    """Computes the average precision metric aver all possible thresholds.

    This metric is an approximation of the :class:`timeeval.metrics.PrAUC`-metric.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments that get passed down to :func:`sklearn.metrics.average_precision_score`

    See Also
    --------
    sklearn.metrics.average_precision_score : Implementation of the average precision metric.
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._kwargs = kwargs

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        score: float = average_precision_score(y_true, y_score, pos_label=1, **self._kwargs)
        return score

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "AVERAGE_PRECISION"


class FScoreAtK(Metric):
    """Computes the F-score at k based on anomaly ranges.

    This metric only considers the top-k predicted anomaly ranges within the scoring by finding a threshold on the
    scoring that produces at least k anomaly ranges.
    If `k` is not specified, the number of anomalies within the ground truth is used as `k`.

    Parameters
    ----------
    k : int (optional)
        Number of top anomalies used to calculate precision. If `k` is not specified (`None`) the number of true
        anomalies (based on the ground truth values) is used.

    See Also
    --------
    timeeval.metrics.thresholding.TopKRangesThresholding : Thresholding approach used.
    """

    def __init__(self, k: Optional[int] = None) -> None:
        self._k = k

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = TopKRangesThresholding(k=self._k).fit_transform(y_true, y_score)
        score: float = ts_fscore(y_true, y_pred, p_alpha=1, r_alpha=1, cardinality="reciprocal")
        return score

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"F_SCORE@K({self._k})"


class PrecisionAtK(Metric):
    """Computes the Precision at k based on anomaly ranges.

    This metric only considers the top-k predicted anomaly ranges within the scoring by finding a threshold on the
    scoring that produces at least k anomaly ranges.
    If `k` is not specified, the number of anomalies within the ground truth is used as `k`.

    Parameters
    ----------
    k : int (optional)
        Number of top anomalies used to calculate precision. If `k` is not specified (`None`) the number of true
        anomalies (based on the ground truth values) is used.

    See Also
    --------
    timeeval.metrics.thresholding.TopKRangesThresholding : Thresholding approach used.
    """

    def __init__(self, k: Optional[int] = None) -> None:
        self._k = k

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = TopKRangesThresholding(k=self._k).fit_transform(y_true, y_score)
        score: float = ts_precision(y_true, y_pred, alpha=1, cardinality="reciprocal")
        return score

    def supports_continuous_scorings(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"PRECISION@K({self._k})"
