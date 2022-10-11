import abc
from typing import Optional, Tuple

import numpy as np


class ThresholdingStrategy(abc.ABC):
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """
    def __int__(self) -> None:
        self.threshold: Optional[float] = None

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Calls :func:`~timeeval.metrics.thresholding.ThresholdingStrategy.find_threshold` to compute and set the
        threshold.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).
        """
        self.threshold = self.find_threshold(y_true, y_score)

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        """Applies the threshold to the anomaly scoring and returns the corresponding binary labels.

        Parameters
        ----------
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.
        """
        return (y_score >= self.threshold).astype(np.int_)

    def fit_transform(self, y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
        """Determines the threshold and applies it to the scoring in one go.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.

        See Also
        --------
        ~timeeval.metrics.thresholding.ThresholdingStrategy.fit : fit-function to determine the threshold.
        ~timeeval.metrics.thresholding.ThresholdingStrategy.transform :
            transform-function to calculate the binary predictions.
        """
        self.fit(y_true, y_score)
        return self.transform(y_score)

    @abc.abstractmethod
    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Abstract method containing the actual code to determine the threshold. Must be overwritten by subclasses!"""
        pass


class NoThresholding(ThresholdingStrategy):
    """Special no-op strategy that checks for already existing binary labels and keeps them untouched. This allows
    applying the metrics on existing binary classification results.
    """

    def fit(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Does nothing (no-op).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).
        """
        pass

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        """Checks if the provided scoring `y_score` is actually a binary classification prediction of integer type. If
        this is the case, the prediction is returned. If not, a :class:`ValueError` is raised.

        Parameters
        ----------
        y_score : np.ndarray
            Anomaly scoring with binary predictions.

        Returns
        -------
        y_pred : np.ndarray
            Array of binary labels; 0 for normal points and 1 for anomalous points.
        """
        if y_score.dtype != np.int_:
            raise ValueError("The NoThresholding strategy can only be used for binary predictions (either 0 or 1). "
                             "Continuous anomaly scorings are not supported, please use any other thresholding "
                             "strategy for this!")
        return y_score

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Does nothing (no-op).

        Parameters
        ----------
        y_true : np.ndarray
            Ignored.
        y_score : np.ndarray
            Ignored.

        Returns
        -------
        None
        """
        pass

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"NoThresholding()"


class FixedValueThresholding(ThresholdingStrategy):
    """Thresholding approach using a fixed threshold value.

    Parameters
    ----------
    threshold : float
        Fixed threshold to use. All anomaly scorings are scaled to the interval [0, 1]
    """
    def __init__(self, threshold: float = 0.8):
        if threshold > 1 or threshold < 0:
            raise ValueError(f"Threshold must be in the interval [0, 1], but was {threshold}!")
        self.threshold = threshold

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Returns the fixed threshold."""
        return self.threshold  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"FixedValueThresholding(threshold={repr(self.threshold)})"


class PercentileThresholding(ThresholdingStrategy):
    """Use the xth-percentile of the anomaly scoring as threshold.

    Parameters
    ----------
    percentile : int
        The percentile of the anomaly scoring to use. Must be between 0 and 100.
    """
    def __init__(self, percentile: int = 90):
        if percentile < 0 or percentile > 100:
            raise ValueError(f"Percentile must be within [0, 100], but was {percentile}!")
        self._percentile = percentile

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes the xth-percentile ignoring NaNs and using a linear interpolation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            The xth-percentile of the anomaly scoring as threshold.
        """
        return np.nanpercentile(y_score, self._percentile)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"PercentileThresholding(percentile={repr(self._percentile)})"


class TopKPointsThresholding(ThresholdingStrategy):
    """Calculates a threshold so that exactly `k` points are marked anomalous.

    Parameters
    ----------
    k : optional int
        Number of expected anomalous points. If `k` is `None`, the ground truth data is used to calculate the real
        number of anomalous points.
    """
    def __init__(self, k: Optional[int] = None):
        if k is not None and k <= 0:
            raise ValueError(f"K must be greater than 0, but was {k}!")
        self._k: Optional[int] = k

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes a threshold based on the number of expected anomalous points.

        The threshold is determined by taking the reciprocal ratio of expected anomalous points to all points as target
        percentile. We, again, ignore NaNs and use a linear interpolation.
        If `k` is `None`, the ground truth data is used to calculate the real ratio of anomalous points to all points.
        Otherwise, `k` is used as the number of expected anomalous points.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold that yields k anomalous points.
        """
        if self._k is None:
            return np.nanpercentile(y_score, (1 - y_true.sum() / y_true.shape[0])*100)  # type: ignore
        else:
            return np.nanpercentile(y_score, (1 - self._k / y_true.shape[0]) * 100)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"TopKPointsThresholding(k={repr(self._k)})"


class TopKRangesThresholding(ThresholdingStrategy):
    """Calculates a threshold so that exactly `k` anomalies are found. The anomalies are either single-points anomalies
    or continuous anomalous ranges.

    Parameters
    ----------
    k : optional int
        Number of expected anomalies. If `k` is `None`, the ground truth data is used to calculate the real number of
        anomalies.
    """
    def __init__(self, k: Optional[int] = None):
        if k is not None and k <= 0:
            raise ValueError(f"K must be greater than 0, but was {k}!")
        self._k: Optional[int] = k

    @staticmethod
    def _count_anomaly_ranges(y_pred: np.ndarray) -> int:
        return int(np.sum(np.diff(np.r_[0, y_pred, 0]) == 1))

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Computes a threshold based on the number of expected anomalous subsequences / ranges (number of anomalies).

        This method iterates over all possible thresholds from high to low to find the first threshold that yields `k`
        or more continuous anomalous ranges.

        If `k` is `None`, the ground truth data is used to calculate the real number of anomalies (anomalous ranges).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold that yields k anomalies.
        """
        if self._k is None:
            self._k = self._count_anomaly_ranges(y_true)
        thresholds: Tuple[float] = np.unique(y_score)[::-1]
        t = thresholds[0]
        y_pred = np.array(y_score >= t, dtype=np.int_)
        # exclude minimum from thresholds, because all points are >= minimum!
        for t in thresholds[1:-1]:
            y_pred = np.array(y_score >= t, dtype=np.int_)
            detected_n = self._count_anomaly_ranges(y_pred)
            if detected_n >= self._k:
                break
        return t

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"TopKRangesThresholding(k={repr(self._k)})"


class SigmaThresholding(ThresholdingStrategy):
    """Computes a threshold :math:`\\theta` based on the anomaly scoring's mean :math:`\mu_s` and the
    standard deviation :math:`\sigma_s`:

    .. math::
       \\theta = \mu_{s} + x \cdot \sigma_{s}

    Parameters
    ----------
    factor: float
        Multiples of the standard deviation to be added to the mean to compute the threshold (:math:`x`).
    """
    def __init__(self, factor: float = 3.0):
        if factor <= 0:
            raise ValueError(f"factor must be greater than 0, but was {factor}!")
        self._factor = factor

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Determines the mean and standard deviation ignoring NaNs of the anomaly scoring and computes the
        threshold using the mentioned equation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Computed threshold based on mean and standard deviation.
        """
        return np.nanmean(y_score) + self._factor * np.nanstd(y_score)  # type: ignore

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"SigmaThresholding(factor={repr(self._factor)})"
