from pathlib import Path
from typing import Union, Any

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class DefaultFactorHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to use the default value multiplied by a factor as parameter value.

    This allows easier specification of parameter search spaces based on the default value. E.g. if we
    consider a n_clusters parameter with default value 50, we can use this heuristic to specify a search space
    of [10, 25, 50, 75, 100] by using the following parameter values:

    - ``"heuristic:DefaultExponentialFactorHeuristic(factor=0.2)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(factor=0.5)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic()"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(factor=1.5)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(factor=2.0)"``

    But if the default parameter value is 100, the search space would be [20, 50, 100, 150, 200].

    Examples
    --------

    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({"window_size": "heuristic:DefaultFactorHeuristic(factor=1, zero_fb=200)"})

    Parameters
    ----------
    factor : float
        Factor to use for the default value. (default: 1.0)
    zero_fb : float
        Value to use for the default value if it is 0. (default: 1.0)
    """
    def __init__(self, factor: float = 1.0, zero_fb: float = 1.0):
        if zero_fb == 0:
            raise ValueError("You cannot supply a zero_fb of 0!")
        self.factor = factor
        self.zero_fb = zero_fb

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        param_name = kwargs["param_name"]
        try:
            default = algorithm.param_schema[param_name]["defaultValue"]
        except KeyError as e:
            raise ValueError(f"Could not find the default value for parameter {param_name}") from e

        if default == 0:
            default = self.zero_fb

        default_type = type(default)
        return default_type(self.factor * default)
