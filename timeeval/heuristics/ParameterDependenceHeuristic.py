import warnings
from pathlib import Path
from typing import Optional, Callable, Any

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class ParameterDependenceHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, source_parameter: str, fn: Optional[Callable[[Any], Any]] = None, factor: Optional[float] = None):
        if fn is not None and factor is not None:
            raise ValueError("You cannot supply a mapping function and a factor at the same time!")
        self.source_parameter = source_parameter
        self.fn = fn
        self.factor = factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Any:
        try:
            original = kwargs["params"][self.source_parameter]
        except KeyError:
            warnings.warn(f"Could not find a value for source parameter '{self.source_parameter}'")
            # don't set the parameter --> use the algorithm default
            return None

        if self.fn is not None:
            return self.fn(original)
        elif self.factor is not None:
            t = type(original)
            return t(self.factor * original)
        else:
            return original
