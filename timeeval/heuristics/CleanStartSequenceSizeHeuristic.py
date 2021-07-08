from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from timeeval.utils.datasets import load_labels_only
from .base import TimeEvalParameterHeuristic


class CleanStartSequenceSizeHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, max_factor: float = 0.1):
        self.max_factor = max_factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        max_size = int(dataset_details.length * self.max_factor)
        labels = load_labels_only(dataset_path)
        return min(max_size, int(labels.argmax()))
