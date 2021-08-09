from pathlib import Path
from typing import Optional, List

import numpy as np

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .PeriodSizeHeuristic import PeriodSizeHeuristic
from .base import TimeEvalParameterHeuristic


class EmbedDimRangeHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, base_factor: float = 1., base_fb_value: int = 50, dim_factors: Optional[List[float]] = None):
        self.base_factor = base_factor
        self.base_fb_value = base_fb_value
        self.dim_factors: List[float] = dim_factors or [0.5, 1.0, 1.5]

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> List[int]:
        heuristic = PeriodSizeHeuristic(factor=self.base_factor, fb_value=self.base_fb_value)
        window_size = heuristic(algorithm, dataset_details, dataset_path)
        return [int(dim) for dim in np.array(self.dim_factors) * window_size]
