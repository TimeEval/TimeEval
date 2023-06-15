import contextlib
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Generator

import numpy as np


class ThresholdingStrategy(ABC):
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """
    def __init__(self) -> None:
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

    @abstractmethod
    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Abstract method containing the actual code to determine the threshold. Must be overwritten by subclasses!"""
        ...


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
        if y_score.dtype not in [np.int_, np.bool_]:
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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
    """Computes a threshold :math:`\\theta` based on the anomaly scoring's mean :math:`\\mu_s` and the
    standard deviation :math:`\\sigma_s`:

    .. math::
       \\theta = \\mu_{s} + x \\cdot \\sigma_{s}

    Parameters
    ----------
    factor: float
        Multiples of the standard deviation to be added to the mean to compute the threshold (:math:`x`).
    """
    def __init__(self, factor: float = 3.0):
        super().__init__()
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


class PyThreshThresholding(ThresholdingStrategy):
    """Uses a thresholder from the `PyThresh <https://github.com/KulikDM/pythresh>`_ package to find a scoring
    threshold and to transform the continuous anomaly scoring into binary anomaly predictions.

    .. warning::
      You need to install PyThresh before you can use this thresholding strategy:

      .. code-block:: bash

        pip install pythresh>=0.2.8

      Please note the additional package requirements for some available thresholders of PyThresh.

    Parameters
    ----------
    pythresh_thresholder : pythresh.thresholds.base.BaseThresholder
        Initiated PyThresh thresholder.
    random_state: Any
        Seed used to seed the numpy random number generator used in some thresholders of PyThresh. Note that PyThresh
        uses the legacy global RNG (``np.random``) and we try to reset the global RNG after calling PyThresh. Can be
        left at its default value for most thresholders that don't use random numbers or provide their own way of
        seeding. Please consult the `PyThresh Documentation <https://pythresh.readthedocs.io/en/latest/index.html>`_
        for details about the individual thresholders.

        .. deprecated:: 1.2.8
            Since pythresh version 0.2.8, thresholders provide a way to set their RNG state correctly. So the parameter
            ``random_state`` is not needed anymore. Please use the pythresh thresholder's parameter to seed it. This
            function's parameter is kept for compatibility with pythresh<0.2.8.

    Examples
    --------
    .. code-block:: python

      from timeeval.metrics.thresholding import PyThreshThresholding
      from pythresh.thresholds.regr import REGR
      import numpy as np

      thresholding = PyThreshThresholding(
          REGR(method="theil")
      )

      y_scores = np.random.default_rng().random(1000)
      y_labels = np.zeros(1000)
      y_pred = thresholding.fit_transform(y_labels, y_scores)
    """

    def __init__(self, pythresh_thresholder: 'BaseThresholder', random_state: Any = None):  # type: ignore
        super().__init__()
        self._thresholder = pythresh_thresholder
        self._predictions: Optional[np.ndarray] = None
        self._random_state: Any = random_state
        if random_state is not None:
            warnings.warn("'random_state' parameter is deprecated. Use pythresh thresholder's parameter instead.",
                          DeprecationWarning,
                          stacklevel=2)

    @staticmethod
    def _make_finite(y_score: np.ndarray) -> np.ndarray:
        """Replaces NaNs with 0 and (Neg)Infs with 1."""
        nan_mask = np.isnan(y_score)
        inf_mask = np.isinf(y_score)
        neginf_mask = np.isneginf(y_score)
        tmp = y_score.copy()
        tmp[nan_mask] = 0
        tmp[inf_mask | neginf_mask] = 1
        return tmp

    def find_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Uses the passed thresholder from the `PyThresh <https://github.com/KulikDM/pythresh>`_ package to determine
        the threshold. Beforehand, the scores are forced to be finite by replacing NaNs with 0 and (Neg)Infs with 1.

        PyThresh thresholders directly compute the binary predictions. Thus, we cache the predictions in the member
        ``_predictions`` and return them when calling
        :func:`~timeeval.metrics.thresholding.PyThreshThresholding.transform`.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_score : np.ndarray
            Anomaly scoring with continuous anomaly scores (same length as `y_true`).

        Returns
        -------
        threshold : float
            Threshold computed by the internal thresholder.
        """
        y_score = self._make_finite(y_score)

        # fix seeding (depending on pythresh version) and if random_state is supplied
        with tmp_np_random_seed_pythresh(self._thresholder, self._random_state):
            # call PyThresh
            self._predictions = self._thresholder.eval(y_score)
            threshold: float = self._thresholder.thresh_

        return threshold

    def transform(self, y_score: np.ndarray) -> np.ndarray:
        if self._predictions is not None:
            return self._predictions
        else:
            return (y_score >= self.threshold).astype(np.int_)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"PyThreshThresholding(pythresh_thresholding={repr(self._thresholder)})"


@contextlib.contextmanager
def tmp_np_random_seed_pythresh(thresholder: 'BaseThresholder', random_state: Any) -> Generator[None, None, None]:   # type: ignore
    import pythresh.version

    pythresh_version = list(map(int, pythresh.version.__version__.split(".")))
    if pythresh_version < [0, 2, 8]:
        # seed legacy np.random for reproducibility
        old_state = np.random.get_state()
        np.random.seed(random_state)
        try:
            yield
        finally:
            # reset np.random state
            np.random.set_state(old_state)
    else:
        if random_state is not None and hasattr(thresholder, "random_state"):
            setattr(thresholder, "random_state", random_state)
        yield
