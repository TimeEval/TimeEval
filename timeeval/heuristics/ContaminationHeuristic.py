from pathlib import Path

import numpy as np

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic
from ..utils.datasets import load_labels_only


class ContaminationHeuristic(TimeEvalParameterHeuristic):
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> float:
        labels = load_labels_only(dataset_path)
        contamination = np.sum(labels) / labels.shape[0]
        return float(contamination)
