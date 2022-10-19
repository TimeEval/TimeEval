from typing import Optional, Tuple

import numpy as np
from prts import ts_precision, ts_recall, ts_fscore

from .auc_metrics import AucMetric
from .metric import Metric
from .thresholding import ThresholdingStrategy, NoThresholding


class RangePrecision(Metric):
    """Computes the range-based precision metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    alpha : float
        Weight of the existence reward. For most - when not all - cases, `p_alpha` should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes).
    """

    def __init__(self, thresholding_strategy: ThresholdingStrategy = NoThresholding(), alpha: float = 0,
                 cardinality: str = "reciprocal", bias: str = "flat", name: str = "RANGE_PRECISION") -> None:
        self._thresholding_strategy = thresholding_strategy
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_precision(y_true, y_pred, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangeRecall(Metric):
    """Computes the range-based recall metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes).
    """

    def __init__(self, thresholding_strategy: ThresholdingStrategy = NoThresholding(), alpha: float = 0,
                 cardinality: str = "reciprocal", bias: str = "flat", name: str = "RANGE_RECALL") -> None:
        self._thresholding_strategy = thresholding_strategy
        self._alpha = alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_recall(y_true, y_pred, alpha=self._alpha, cardinality=self._cardinality, bias=self._bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangeFScore(Metric):
    """Computes the range-based F-score using the recall and precision metrics by Tatbul et al. at NeurIPS 2018
    [TatbulEtAl2018]_.

    The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its
    worst value at 0. This implementation uses the range-based precision and range-based recall as basis.

    Parameters
    ----------
    thresholding_strategy : ThresholdingStrategy
        Strategy used to find a threshold over continuous anomaly scores to get binary labels.
        Use :class:`timeeval.metrics.thresholding.NoThresholding` for results that already contain binary labels.
    beta : float
        F-score beta determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    r_alpha : float
        Weight of the existence reward. If 0: no existence reward, if 1: only existence reward.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    p_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    r_bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    name : str
        Custom name for this metric (e.g. including your parameter changes). If `None`, will include the beta-value in
        the name: "RANGE_F{beta}_SCORE".
    """

    def __init__(self,
                 thresholding_strategy: ThresholdingStrategy = NoThresholding(),
                 beta: float = 1,
                 p_alpha: float = 0,
                 r_alpha: float = 0.5,
                 cardinality: str = "reciprocal",
                 p_bias: str = "flat",
                 r_bias: str = "flat",
                 name: Optional[str] = None) -> None:
        self._thresholding_strategy = thresholding_strategy
        self._beta = beta
        self._p_alpha = p_alpha
        self._r_alpha = r_alpha
        self._cardinality = cardinality
        self._p_bias = p_bias
        self._r_bias = r_bias
        self._name = f"RANGE_F{self._beta:.2f}_SCORE" if name is None else name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_pred = self._thresholding_strategy.fit_transform(y_true, y_score)
        score: float = ts_fscore(y_true, y_pred,
                                 beta=self._beta,
                                 p_alpha=self._p_alpha, r_alpha=self._r_alpha,
                                 cardinality=self._cardinality,
                                 p_bias=self._p_bias, r_bias=self._p_bias)
        return score

    def supports_continuous_scorings(self) -> bool:
        return not isinstance(self._thresholding_strategy, NoThresholding)

    @property
    def name(self) -> str:
        return self._name


class RangePrecisionRangeRecallAUC(AucMetric):
    """Computes the area under the precision recall curve when using the range-based precision and range-based
    recall metric introduced by Tatbul et al. at NeurIPS 2018 [TatbulEtAl2018]_.

    Parameters
    ----------
    max_samples: int
        TimeEval uses a community implementation of the range-based precision and recall metrics, which is quite slow.
        To prevent long runtimes caused by scorings with high precision (many thresholds), just a specific amount of
        possible thresholds is sampled. This parameter controls the maximum number of thresholds; too low numbers
        degrade the metrics' quality.
    r_alpha : float
        Weight of the existence reward for the range-based recall.
    p_alpha : float
        Weight of the existence reward for the range-based precision. For most - when not all - cases, `p_alpha`
        should be set to 0.
    cardinality : {'reciprocal', 'one', 'udf_gamma'}
        Cardinality type.
    bias : {'flat', 'front', 'middle', 'back'}
        Positional bias type.
    plot : bool
    plot_store : bool
    name : str
        Custom name for this metric (e.g. including your parameter changes).


    .. rubric:: References

    .. [TatbulEtAl2018] Tatbul, Nesime, Tae Jun Lee, Stan Zdonik, Mejbah Alam, and Justin Gottschlich. "Precision and Recall for
       Time Series." In Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS),
       1920–30. 2018. http://papers.nips.cc/paper/7462-precision-and-recall-for-time-series.pdf.
    """

    def __init__(self, max_samples: int = 50, r_alpha: float = 0.5, p_alpha: float = 0, cardinality: str = "reciprocal",
                 bias: str = "flat", plot: bool = False, plot_store: bool = False, name: str = "RANGE_PR_AUC") -> None:
        super().__init__(plot, plot_store)
        self._max_samples = max_samples
        self._r_alpha = r_alpha
        self._p_alpha = p_alpha
        self._cardinality = cardinality
        self._bias = bias
        self._name = name

    def score(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return self._auc(y_true, y_score, self._range_precision_recall_curve)

    def _range_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        thresholds = np.unique(y_score)
        thresholds.sort()
        # The first precision and recall values are precision=class balance and recall=1.0, which corresponds to a
        # classifier that always predicts the positive class, independently of the threshold. This means that we can
        # skip the first threshold!
        p0 = y_true.sum() / len(y_true)
        r0 = 1.0
        thresholds = thresholds[1:]

        # sample thresholds
        n_thresholds = thresholds.shape[0]
        if n_thresholds > self._max_samples:
            every_nth = n_thresholds // (self._max_samples - 1)
            sampled_thresholds = thresholds[::every_nth]
            if thresholds[-1] == sampled_thresholds[-1]:
                thresholds = sampled_thresholds
            else:
                thresholds = np.r_[sampled_thresholds, thresholds[-1]]

        recalls = np.zeros_like(thresholds)
        precisions = np.zeros_like(thresholds)
        for i, threshold in enumerate(thresholds):
            y_pred = (y_score >= threshold).astype(np.int64)
            recalls[i] = ts_recall(y_true, y_pred,
                                   alpha=self._r_alpha,
                                   cardinality=self._cardinality,
                                   bias=self._bias)
            precisions[i] = ts_precision(y_true, y_pred,
                                         alpha=self._p_alpha,
                                         cardinality=self._cardinality,
                                         bias=self._bias)
        # first sort by recall, then by precision to break ties (important for noisy scorings)
        sorted_idx = np.lexsort((precisions * (-1), recalls))[::-1]
        return np.r_[p0, precisions[sorted_idx], 1], np.r_[r0, recalls[sorted_idx], 0], thresholds

    @property
    def name(self) -> str:
        return self._name
