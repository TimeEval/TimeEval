"""
This module contains all metrics that can be used with TimeEval. The metrics are divided into five different categories:

- **Classification-metrics:** These metrics are defined over binary classification predictions (zeros or ones), thus
  they require a thresholding strategy to convert anomaly scorings to binary classification results.

    - :class:`~timeeval.metrics.Precision`
    - :class:`~timeeval.metrics.Recall`
    - :class:`~timeeval.metrics.F1Score`

- **AUC-metrics:** All AUC-Metrics support continuous scorings, and calculate the area under a custom curve function.

    - :class:`~timeeval.metrics.RocAUC`
    - :class:`~timeeval.metrics.PrAUC`

- **Range-metrics:** Range-metrics compute the quality scores for anomaly ranges (windows) instead of each point in the
  time series.

    - :class:`~timeeval.metrics.RangePrecision`
    - :class:`~timeeval.metrics.RangeRecall`
    - :class:`~timeeval.metrics.RangeFScore`
    - :class:`~timeeval.metrics.RangePrecisionRangeRecallAUC`

- **VUS-metrics:** The metrics of this category share a custom definition of range-based recall and range-based
  precision [PaparrizosEtAl2022]_.

    - :class:`~timeeval.metrics.RangePrAUC`
    - :class:`~timeeval.metrics.RangeRocAUC`
    - :class:`~timeeval.metrics.RangePrVUS`
    - :class:`~timeeval.metrics.RangeRocVUS`

- **Other-metrics:** Metrics that don't belong to any of the above categories:

    - :class:`~timeeval.metrics.AveragePrecision`
    - :class:`~timeeval.metrics.PrecisionAtK`
    - :class:`~timeeval.metrics.FScoreAtK`

All metrics inherit from the abstract base class :class:`~timeeval.metrics.Metric`, and implement the ``__call__``
method, the ``supports_continuous_scorings`` method, and the ``name`` property. This allows them to be used within
TimeEval and on their own. You can also implement your own metrics by inheriting from :class:`timeeval.metrics.Metric`
(see its documentation for more information).

Examples
--------

Using the default metric list that just contains ROC_AUC:

>>> from timeeval import TimeEval, DefaultMetrics
>>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
>>>          metrics=DefaultMetrics.default_list())

Using a custom selection of metrics:

>>> from timeeval import TimeEval
>>> from timeeval.metrics import RangeRocAUC, RangeRocVUS, RangePrAUC, RangePrVUS
>>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
>>>          metrics=[RangeRocAUC(buffer_size=100), RangeRocVUS(max_buffer_size=100),
>>>                  RangePrAUC(buffer_size=100), RangePrVUS(max_buffer_size=100)])

Using the metrics without TimeEval:

>>> import numpy as np
>>> from timeeval import DefaultMetrics
>>> from timeeval.metrics import RangePrAUC
>>> from timeeval.metrics.thresholding import PercentileThresholding
>>> rng = np.random.default_rng(42)
>>> y_true = rng.random(100) > 0.5
>>> y_score = rng.random(100)
>>> metrics = [
>>>     # default metrics are already parameterized objects:
>>>     DefaultMetrics.ROC_AUC,
>>>     # all metrics (in general) are classes that need to be instantiated with their parameterization:
>>>     RangePrAUC(buffer_size=100),
>>>     # classification metrics need a thresholding strategy for continuous scorings:
>>>     F1Score(PercentileThresholding(percentile=95))
>>> ]
>>> # compute the metrics
>>> for m in metrics:
>>>     metric_value = m(y_true, y_score)
>>>     print(f"{m.name} = {metric_value}")
"""
from typing import List

from .auc_metrics import AucMetric, RocAUC, PrAUC
from .classification_metrics import Precision, Recall, F1Score
from .metric import Metric
from .other_metrics import AveragePrecision, PrecisionAtK, FScoreAtK
from .range_metrics import RangePrecisionRangeRecallAUC, RangePrecision, RangeRecall, RangeFScore
from .vus_metrics import RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS


class DefaultMetrics:
    """Default metrics of TimeEval that can be used directly for time series anomaly detection algorithms without
    further configuration.

    Examples
    --------
    Using the default metric list that just contains ROC_AUC:

    >>> from timeeval import TimeEval, DefaultMetrics
    >>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
    >>>          metrics=DefaultMetrics.default_list())

    You can also specify multiple default metrics:

    >>> from timeeval import TimeEval, DefaultMetrics
    >>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
    >>>          metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC])
    """

    ROC_AUC = RocAUC()
    PR_AUC = PrAUC()
    RANGE_PR_AUC = RangePrecisionRangeRecallAUC(max_samples=50, r_alpha=0, cardinality="one", bias="flat")
    AVERAGE_PRECISION = AveragePrecision()
    RANGE_PRECISION = RangePrecision()
    RANGE_RECALL = RangeRecall()
    RANGE_F1 = RangeFScore(beta=1)
    FIXED_RANGE_PR_AUC = RangePrecisionRangeRecallAUC(name="FIXED_RANGE_PR_AUC")

    @staticmethod
    def default() -> Metric:
        """TimeEval's default metric ROC_AUC."""
        return DefaultMetrics.ROC_AUC

    @staticmethod
    def default_list() -> List[Metric]:
        """The list containing TimeEval's single default metric ROC_AUC. For your convenience and usage as default
        parameter in many TimeEval library functions."""
        return [DefaultMetrics.ROC_AUC]
