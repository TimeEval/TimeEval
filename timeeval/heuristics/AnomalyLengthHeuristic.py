from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TimeEvalParameterHeuristic


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from pathlib import Path
    from ..algorithm import Algorithm
    from ..datasets import Dataset


class AnomalyLengthHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, agg_type: str = "median"):
        if agg_type not in ["min", "median", "max"]:
            raise ValueError(f"'agg_type' must be one of min, median, or max. But '{agg_type}' was given.")
        self.agg_type = agg_type

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:  # type: ignore[no-untyped-def]
        if self.agg_type == "min":
            value = dataset_details.min_anomaly_length
        elif self.agg_type == "max":
            value = dataset_details.max_anomaly_length
        else:  # self.agg_type == "median"
            value = dataset_details.median_anomaly_length
        return int(value)
