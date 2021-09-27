from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class DefaultExponentialFactorHeuristic(TimeEvalParameterHeuristic):
    def __init__(self, exponent: int = 0):
        self.exponent = exponent

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        param_name = kwargs["param_name"]
        try:
            default = algorithm.params[param_name]["defaultValue"]
        except KeyError as e:
            raise ValueError(f"Could not find the default value for parameter {param_name}") from e

        return 10**self.exponent * default
