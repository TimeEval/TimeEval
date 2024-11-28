import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.utils import column_or_1d, assert_all_finite, check_consistent_length


class Metric(ABC):
    """Base class for metric implementations that score anomaly scorings against ground truth binary labels. Every
    subclass must implement :func:`~timeeval.metrics.Metric.name`, :func:`~timeeval.metrics.Metric.score`, and
    :func:`~timeeval.metrics.Metric.supports_continuous_scorings`.

    Examples
    --------
    You can implement a new TimeEval metric easily by inheriting from this base class. A simple metric, for example,
    uses a fixed threshold to get binary labels and computes the false positive rate:

    >>> from timeeval.metrics import Metric
    >>> class FPR(Metric):
    >>>     def __init__(self, threshold: float = 0.8):
    >>>         self._threshold = threshold
    >>>     @property
    >>>     def name(self) -> str:
    >>>         return f"FPR@{self._threshold}"
    >>>     def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
    >>>         y_pred = y_score >= self._threshold
    >>>         fp = np.sum(y_pred & ~y_true)
    >>>         return fp / (fp + np.sum(y_true))
    >>>     def supports_continuous_scorings(self) -> bool:
    >>>         return True

    This metric can then be used in TimeEval:

    >>> from timeeval import TimeEval
    >>> from timeeval.metrics import DefaultMetrics
    >>> timeeval = TimeEval(dmgr=..., datasets=[], algorithms=[],
    >>>                     metrics=[FPR(threshold=0.8), DefaultMetrics.ROC_AUC])
    """

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:  # type: ignore[no-untyped-def]
        y_true, y_score = self._validate_scores(y_true, y_score, **kwargs)
        if np.unique(y_score).shape[0] == 1:
            warnings.warn("Cannot compute metric for a constant value in y_score, returning 0.0!")
            return 0.
        return self.score(y_true, y_score)

    def _validate_scores(self, y_true: np.ndarray, y_score: np.ndarray,
                         inf_is_1: bool = True,
                         neginf_is_0: bool = True,
                         nan_is_0: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.array(y_true).copy()
        y_score = np.array(y_score).copy()
        # check labels
        if self.supports_continuous_scorings() and y_true.dtype == np.float_ and y_score.dtype == np.int_:
            warnings.warn("Assuming that y_true and y_score where permuted, because their dtypes indicate so. "
                          "y_true should be an integer array and y_score a float array!")
            return self._validate_scores(y_score, y_true)

        y_true: np.ndarray = column_or_1d(y_true)  # type: ignore
        assert_all_finite(y_true)

        # check scores
        y_score: np.ndarray = column_or_1d(y_score)  # type: ignore

        check_consistent_length([y_true, y_score])
        if not self.supports_continuous_scorings():
            if y_score.dtype not in [np.int_, np.bool_]:
                raise ValueError(
                    "When using Metrics other than AUC-metric that need discrete (0 or 1) scores (like "
                    "Precision, Recall or F1-Score), the scores must be integers and should only contain "
                    "the values {0, 1}. Please consider applying a threshold to the scores!"
                )
        else:
            if not np.issubdtype(y_score.dtype, np.floating):
                raise ValueError(
                    f"When using continuous scoring metrics, the scores must be floats!. "
                    f"Got {y_score.dtype} instead."
                )

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
            y_score[nan_mask] = 0
        else:
            penalize_mask = penalize_mask | nan_mask
        y_score[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(np.int_)

        assert_all_finite(y_score)
        return y_true, y_score

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of this metric."""
        ...

    @abstractmethod
    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Implementation of the metric's scoring function.

        Please use :func:`~timeeval.metrics.Metric.__call__` instead of calling this function directly!

        Examples
        --------

        Instantiate a metric and call it using the ``__call__`` method:

        >>> import numpy as np
        >>> from timeeval.metrics import RocAUC
        >>> metric = RocAUC(plot=False)
        >>> metric(np.array([0, 1, 1, 0]), np.array([0.1, 0.4, 0.35, 0.8]))
        0.5

        """
        ...

    @abstractmethod
    def supports_continuous_scorings(self) -> bool:
        """Whether this metric accepts continuous anomaly scorings as input (``True``) or binary classification
        labels (``False``)."""
        ...
