import abc

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from .metric import Metric
from .thresholding import ThresholdingStrategy, NoThresholding


class ClassificationMetric(Metric, abc.ABC):
    """Base class for standard classification metrics.

    All classification metrics are defined over binary classification predictions (zeros or ones), thus all of them
    require a thresholding strategy to convert anomaly scorings to binary classification results. The thresholding
    strategy can be set using the
    """

    def __init__(self, thresholding_strategy: ThresholdingStrategy):
        self._thresholding_strategy = thresholding_strategy

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        return self._internal_score(y_true, y_pred)

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @abc.abstractmethod
    def _internal_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Classification metrics should implement this method to compute the metric score."""
        pass


class Precision(ClassificationMetric):
    """Computes the precision metric.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Thresholding strategy used to transform the anomaly scorings to binary classification predictions.

    See Also
    --------
    sklearn.metrics.precision_score : Implementation of the precision metric.
    """
    def __init__(self, thresholding_strategy: ThresholdingStrategy):
        super().__init__(thresholding_strategy)

    def _internal_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred)  # type: ignore

    @property
    def name(self) -> str:
        return f"Precision_{self._thresholding_strategy}"


class Recall(ClassificationMetric):
    """Computes the recall metric.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Thresholding strategy used to transform the anomaly scorings to binary classification predictions.

    See Also
    --------
    sklearn.metrics.recall_score : Implementation of the recall metric.
    """
    def __init__(self, thresholding_strategy: ThresholdingStrategy):
        super().__init__(thresholding_strategy)

    def _internal_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred)  # type: ignore

    @property
    def name(self) -> str:
        return f"Recall_{self._thresholding_strategy}"


class F1Score(ClassificationMetric):
    """Computes the F1 metric, which is the harmonic mean of precision and recall.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Thresholding strategy used to transform the anomaly scorings to binary classification predictions.

    See Also
    --------
    sklearn.metrics.f1_score : Implementation of the F1 metric.
    """
    def __init__(self, thresholding_strategy: ThresholdingStrategy):
        super().__init__(thresholding_strategy)

    def _internal_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred)  # type: ignore

    @property
    def name(self) -> str:
        return f"F1Score_{self._thresholding_strategy}"
