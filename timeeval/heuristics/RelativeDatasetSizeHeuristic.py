from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class RelativeDatasetSizeHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return int(dataset_details.length * self.factor)
