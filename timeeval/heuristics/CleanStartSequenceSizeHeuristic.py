from __future__ import annotations

from typing import TYPE_CHECKING

from timeeval.utils.datasets import load_labels_only
from .base import TimeEvalParameterHeuristic


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from pathlib import Path
    from ..algorithm import Algorithm
    from ..datasets import Dataset


class CleanStartSequenceSizeHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, max_factor: float = 0.1):
        self.max_factor = max_factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:  # type: ignore[no-untyped-def]
        max_size = int(dataset_details.length * self.max_factor)
        labels = load_labels_only(dataset_path)
        return min(max_size, int(labels.argmax()))
