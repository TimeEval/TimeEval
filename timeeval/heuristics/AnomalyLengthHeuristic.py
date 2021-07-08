from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class AnomalyLengthHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, agg_type: str = "median"):
        if agg_type not in ["min", "median", "max"]:
            raise ValueError(f"'agg_type' must be one of min, median, or max. But '{agg_type}' was given.")
        self.agg_type = agg_type

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        if self.agg_type == "min":
            value = dataset_details.min_anomaly_length
        elif self.agg_type == "max":
            value = dataset_details.max_anomaly_length
        else:  # self.agg_type == "median"
            value = dataset_details.median_anomaly_length
        return int(value)
