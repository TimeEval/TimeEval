import warnings
from pathlib import Path
from typing import Optional, Callable, Any

from timeeval import Algorithm
from timeeval.datasets import Dataset
from .base import TimeEvalParameterHeuristic


class ParameterDependenceHeuristic(TimeEvalParameterHeuristic):
    """Heuristic to use the value of another parameter as parameter value.

    ``ParameterDependenceHeuristic`` can be used to create a parameter value that depends on another parameter. This can
    be done by either supplying a mapping function or a factor. If a mapping function is supplied, it is called with the
    value of the source parameter as the only argument. If a factor is supplied, the value of the source parameter is
    multiplied by the factor. **You cannot supply both a mapping function and a factor!** This heuristic is evaluated
    after all other heuristics, so you can use it to create a parameter value that depends on the values of other
    parameters filled by heuristics.

    Examples
    --------
    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({
    ...     "latent_dim": "heuristic:ParameterDependenceHeuristic(source_parameter='window_size', factor=0.5)"
    ... })

    >>> from timeeval.params import FixedParameters
    >>> params = FixedParameters({
    ...     "latent_dims": "heuristic:ParameterDependenceHeuristic(source_parameter='window_size', fn=lambda x: [x // 2, x, x * 2])"
    ... })

    Parameters
    ----------
    source_parameter : str
        Name of the parameter to use as source.
    fn : Callable[[Any], Any], optional
        Mapping function to use on the source parameter value. (default: None)
    factor : float, optional
        Factor to multiply the source parameter value with. (default: None)
    """
    def __init__(self, source_parameter: str, fn: Optional[Callable[[Any], Any]] = None, factor: Optional[float] = None):
        if fn is not None and factor is not None:
            raise ValueError("You cannot supply a mapping function and a factor at the same time!")
        self.source_parameter = source_parameter
        self.fn = fn
        self.factor = factor

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Any:  # type: ignore[no-untyped-def]
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
