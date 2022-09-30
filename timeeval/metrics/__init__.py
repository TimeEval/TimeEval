from typing import List

from .auc_metrics import RocAUC, PrAUC
from .metric import Metric
from .other_metrics import AveragePrecision, PrecisionAtK, FScoreAtK
from .range_metrics import RangePrAUC, RangePrecision, RangeRecall, RangeFScore


class DefaultMetrics:
    ROC_AUC = RocAUC()
    PR_AUC = PrAUC()
    RANGE_PR_AUC = RangePrAUC(max_samples=50, r_alpha=0, cardinality="one", bias="flat")
    AVERAGE_PRECISION = AveragePrecision()
    RANGE_PRECISION = RangePrecision()
    RANGE_RECALL = RangeRecall()
    RANGE_F1 = RangeFScore(beta=1)
    FIXED_RANGE_PR_AUC = RangePrAUC(name="FIXED_RANGE_PR_AUC")

    @staticmethod
    def default() -> Metric:
        return DefaultMetrics.ROC_AUC

    @staticmethod
    def default_list() -> List[Metric]:
        return [DefaultMetrics.ROC_AUC]
