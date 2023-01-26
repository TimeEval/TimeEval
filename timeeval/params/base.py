from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sized, Iterator, Mapping

from .params import FixedParams


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from ..algorithm import Algorithm
    from ..datasets import Dataset
    from .params import Params
    from typing import Any
    from .params import Params


class ParameterConfig(abc.ABC, Sized):
    """Base class for algorithm hyperparameter configurations.

    Currently, TimeEval supports three kinds of parameter configurations:

    1. :class:`~timeeval.params.FixedParameters`: A single parameter setting with one value for each parameter.
    2. :class:`~timeeval.params.IndependentParameterGrid` and :class:`~timeeval.params.FullParameterGrid`: Parameter
       search using the specification of a parameter grid, where each parameter can have multiple values. Depending on
       the parameter grid, TimeEval will build a parameter search space and test all combinations of parameters.
    3. :class:`~timeeval.params.BayesianParameterSearch`: Parameter search using Bayesian optimization.

    """

    @abc.abstractmethod
    def iter(self, algorithm: Algorithm, dataset: Dataset) -> Iterator[Params]:
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over Params
            Yields a params object that maps each parameter to a single value.
        """
        ...

    @staticmethod
    def defaults() -> ParameterConfig:
        """Returns the default parameter configuration that has only a single parameter setting with no parameters."""
        return FixedParameters({})


class FixedParameters(ParameterConfig):
    """Single parameters setting with one value for each.

    Iterating over this grid yields the input setting as the first and only element.

    Parameters
    ----------
    params : dict of str to Any
        The parameter setting to be evaluated, as a dictionary mapping parameters to allowed values.
        An empty dict signifies default parameters.

    Examples
    --------
    >>> from timeeval.params import FixedParameters
    >>> params = {"a": 2, "b": True}
    >>> list(FixedParameters(params)) == (
    ...    [{"a": 2, "b": True}])
    True
    >>> FixedParameters(params)[0] == {"a": 2, "b": True}
    True
    """

    def __init__(self, params: Mapping[str, Any]):
        if not isinstance(params, Mapping):
            if isinstance(params, (list, Iterator)):
                raise TypeError("A sequence of grids (Iterable[Mapping[str, Any]) is not supported by this "
                                f"ParameterConfig ({params}). Please use a "
                                "`timeeval.search.IndependentParameterGrid` for this!")
            else:
                raise TypeError(f"Parameters are not provided as a dict ({params})")
        self._params = [FixedParams(params)]

    def iter(self, algorithm: Algorithm, dataset: Dataset) -> Iterator[Params]:
        return self.__iter__()

    def __iter__(self) -> Iterator[Params]:
        return iter(self._params)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i: int) -> Params:
        assert i == 0, "FixedParameters only has one element"
        return self._params[i]
