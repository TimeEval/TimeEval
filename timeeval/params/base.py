from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Sized, Mapping, Iterator

from .params import FixedParams


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from typing import Any
    from ..algorithm import Algorithm
    from ..datasets import Dataset
    from .params import Params


class ParameterConfig(abc.ABC, Sized):
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
        return FixedParameters({})


class FixedParameters(ParameterConfig):
    """Single parameters setting with one value for each.

    Iterating over this grid yields the input setting as the first and only element. Uses the
    sklearn.model_selection.ParameterGrid internally.

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

    See also
    --------
    :class:`sklearn.model_selection.ParameterGrid`:
        Used internally to represent the parameter grids.
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