from pathlib import Path
from typing import Optional, List

import numpy as np

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .PeriodSizeHeuristic import PeriodSizeHeuristic
from .base import TimeEvalParameterHeuristic


class EmbedDimRangeHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to use a range of embedding dimensions as parameter value.

    The base dimensionality is calculated based on the :class:`~timeeval.heuristics.PeriodSizeHeuristic`, base factor,
    and base fallback value. The base dimensionality is then multiplied by the factors specified in ``dim_factors`` to
    create the embedding dimension range.

    Examples
    --------
    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({
    ...     "embed_dim": "heuristic:EmbedDimRangeHeuristic(base_factor=1, base_fb_value=50, dim_factors=[0.5, 1.0, 1.5])"
    ... })

    Parameters
    ----------
    base_factor : float
        Factor to use for the base dimensionality. Directly passed on to the ``PeriodSizeHeuristic``. (default: 1.0)
    base_fb_value : int
        Fallback value to use for the base dimensionality. Directly passed on to the ``PeriodSizeHeuristic``.
        (default: 50)
    dim_factors : List[float]
        Factors to use for the creation of the embedding dimension range. (default: [0.5, 1.0, 1.5])
    """
    def __init__(self, base_factor: float = 1., base_fb_value: int = 50, dim_factors: Optional[List[float]] = None):
        self.base_factor = base_factor
        self.base_fb_value = base_fb_value
        self.dim_factors: List[float] = dim_factors or [0.5, 1.0, 1.5]

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> List[int]:  # type: ignore[no-untyped-def]
        heuristic = PeriodSizeHeuristic(factor=self.base_factor, fb_value=self.base_fb_value)
        window_size = heuristic(algorithm, dataset_details, dataset_path)
        return [int(dim) for dim in np.array(self.dim_factors) * window_size]
