from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import TimeEvalParameterHeuristic
from ..utils.datasets import load_labels_only


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from pathlib import Path
    from ..algorithm import Algorithm
    from ..datasets import Dataset


class ContaminationHeuristic(TimeEvalParameterHeuristic):
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> float:  # type: ignore[no-untyped-def]
        labels = load_labels_only(dataset_path)
        contamination = np.sum(labels) / labels.shape[0]
        return float(contamination)
