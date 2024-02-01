import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .AnomalyLengthHeuristic import AnomalyLengthHeuristic
from .base import TimeEvalParameterHeuristic


def _is_none(period: Optional[int]) -> bool:
    return period is None or np.isnan(period)


_is_still_none = _is_none  # syntactic sugar ;)


class PeriodSizeHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to use the period size of the dataset as parameter value.

    Not all datasets have a period size, so this heuristic uses the following fallbacks in order:

    1. If ``fb_anomaly_length_agg_type`` is specified, the :class:`~timeeval.heuristics.AnomalyLengthHeuristic`
    with the specified aggregation type is used as fallback.
    2. If ``fb_value`` is specified, it is directly used as fallback.

    Examples
    --------
    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({
    ...     "window_size": "heuristic:PeriodSizeHeuristic(factor=1.0, fb_anomaly_length_agg_type='median', fb_value=100)"
    ... })

    Parameters
    ----------
    factor : float
        Factor to use for the period size. (default: 1.0)
    fb_anomaly_length_agg_type : str, optional
        Aggregation type to use for the :class:`~timeeval.heuristics.AnomalyLengthHeuristic` fallback. (default: None)
    fb_value : int, optional
        Value to use as fallback if no period size is available. (default: 1)
    """
    def __init__(self, factor: float = 1., fb_anomaly_length_agg_type: Optional[str] = None, fb_value: int = 1):
        self.factor = factor
        self.fb_anomaly_length_agg_type = fb_anomaly_length_agg_type
        self.fb_value = fb_value

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:  # type: ignore[no-untyped-def]
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
