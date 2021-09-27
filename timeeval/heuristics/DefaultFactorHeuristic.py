from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class DefaultFactorHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        param_name = kwargs["param_name"]
        try:
            default = algorithm.params[param_name]["defaultValue"]
        except KeyError as e:
            raise ValueError(f"Could not find the default value for parameter {param_name}") from e

        return self.factor * default
