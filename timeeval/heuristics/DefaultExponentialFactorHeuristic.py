from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TimeEvalParameterHeuristic


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from typing import Any
    from pathlib import Path
    from ..algorithm import Algorithm
    from ..datasets import Dataset


class DefaultExponentialFactorHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to use the default value multiplied by a factor of $10^{exponent}$ as parameter value.

    This allows easier specification of exponential parameter search spaces based on the default value. E.g. if we
    consider a learning rate parameter with default value 0.01, we can use this heuristic to specify a search space
    of [0.0001, 0.001, 0.01, 0.1, 1] by using the following parameter values:

    - ``"heuristic:DefaultExponentialFactorHeuristic(exponent=-2)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(exponent=-1)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic()"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(exponent=1)"``
    - ``"heuristic:DefaultExponentialFactorHeuristic(exponent=2)"``

    But if the default parameter value is 0.5, the search space would be [0.005, 0.05, 0.5, 5, 50].

    Examples
    --------

    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({"window_size": "heuristic:DefaultExponentialFactorHeuristic(exponent=1, zero_fb=200)"})

    Parameters
    ----------
    exponent : int
        Exponent to use for the factor. (default: 0)
    zero_fb : float
        Value to use for the default value if it is 0. (default: 1.0)
    """
    def __init__(self, exponent: int = 0, zero_fb: float = 1.0):
        if zero_fb == 0:
            raise ValueError("You cannot supply a zero_fb of 0!")
        self.exponent = exponent
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
        return default_type(10**self.exponent * default)
