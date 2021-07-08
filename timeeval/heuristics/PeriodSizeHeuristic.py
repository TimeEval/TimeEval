import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .AnomalyLengthHeuristic import AnomalyLengthHeuristic
from .base import TimeEvalParameterHeuristic


def _is_none(period) -> bool:
    return period is None or np.isnan(period)


_is_still_none = _is_none  # syntactic sugar ;)


class PeriodSizeHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, factor: float = 1., fb_anomaly_length_agg_type: Optional[str] = None, fb_value: int = 1):
        self.factor = factor
        self.fb_anomaly_length_agg_type = fb_anomaly_length_agg_type
        self.fb_value = fb_value

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        period = dataset_details.period_size
        if _is_none(period) and self.fb_anomaly_length_agg_type is not None:
            try:
                anomaly_length_heuristic = AnomalyLengthHeuristic(agg_type=self.fb_anomaly_length_agg_type)
                period = anomaly_length_heuristic(algorithm, dataset_details, dataset_path)
                warnings.warn(f"{algorithm.name}: No period_size specified for dataset {dataset_details.datasetId}. "
                              f"Using AnomalyLengthHeuristic({self.fb_anomaly_length_agg_type}) as fallback.")
            except ValueError:
                pass

        if _is_still_none(period):
            warnings.warn(f"{algorithm.name}: No period_size specified for dataset {dataset_details.datasetId}. "
                          f"Using fixed value '{self.fb_value}' as parameter value.")
            return self.fb_value
        # period: Optional[int] but _is_none guards for None, so it's an int!
        return int(period * self.factor)  # type: ignore
