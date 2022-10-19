from typing import List

from .auc_metrics import AucMetric, RocAUC, PrAUC
from .classification_metrics import Precision, Recall, F1Score
from .metric import Metric
from .other_metrics import AveragePrecision, PrecisionAtK, FScoreAtK
from .range_metrics import RangePrecisionRangeRecallAUC, RangePrecision, RangeRecall, RangeFScore
from .thresholding import PercentileThresholding
from .vus_metrics import RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS


class DefaultMetrics:
    """Default metrics of TimeEval that can be used directly for time series anomaly detection algorithms without
    further configuration.

    Examples
    --------
    Using the default metric list that just contains ROC_AUC:

    >>> from timeeval import TimeEval, DefaultMetrics
    >>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
    >>>          metric=DefaultMetrics.default_list())

    You can also specify multiple default metrics:

    >>> from timeeval import TimeEval, DefaultMetrics
    >>> TimeEval(dataset_mgr=..., datasets=[], algorithms=[],
    >>>          metric=[DefaultMetrics.ROC_AUC, DefaultMetrics.PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC])
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
