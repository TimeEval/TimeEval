from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class RelativeDatasetSizeHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to set a parameter value depending on the size of the dataset (length of the time series).

    Examples
    --------
    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({"n_init": "heuristic:RelativeDatasetSizeHeuristic(factor=0.1)"})

    Parameters
    ----------
    factor : float
        Factor to multiply the dataset length with to get the parameter value. (default: 0.1)
    """
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:  # type: ignore[no-untyped-def]
        return int(dataset_details.length * self.factor)
