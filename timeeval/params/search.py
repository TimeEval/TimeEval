import abc
from collections.abc import Iterable, Sized
from typing import Iterator, Any, Dict, Mapping, Optional

from sklearn.model_selection import ParameterGrid


class ParameterConfig(abc.ABC, Iterable, Sized):
    @property
    @abc.abstractmethod
    def param_grid(self) -> ParameterGrid:
        """The parameter search grid.

        Returns
        -------
        param_grid: sklearn parameter grid object
            A parameter search grid compatible with sklearn: :class:`sklearn.model_selection.ParameterGrid`
        """
        pass

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of str to any
            Yields dictionaries mapping each parameter to one of its allowed values.
        """
        return iter(self.param_grid)

    def __len__(self) -> int:
        """Number of points on the grid."""
        return len(self.param_grid)

    def __getitem__(self, i) -> Dict[str, Any]:
        return self.param_grid[i]

    @staticmethod
    def defaults() -> "ParameterConfig":
        return FixedParameters({})


class FullParameterGrid(ParameterConfig):
    """Grid of parameters with a discrete number of values for each.

    Iterating over this grid yields the full cartesian product of all available parameter combinations. Uses the
    sklearn.model_selection.ParameterGrid internally.

    Parameters
    ----------
    param_grid : dict of str to sequence
        The parameter grid to explore, as a dictionary mapping parameters to sequences of allowed values.
        An empty dict signifies default parameters.

    Examples
    --------
    >>> from timeeval.params.search import FullParameterGrid
    >>> params = {"a": [1, 2], "b": [True, False]}
    >>> list(FullParameterGrid(params)) == (
    ...    [{"a": 1, "b": True}, {"a": 1, "b": False},
    ...     {"a": 2, "b": True}, {"a": 2, "b": False}])
    True
    >>> FullParameterGrid(params)[1] == {"a": 1, "b": False}
    True

    See also
    --------
    :class:`sklearn.model_selection.ParameterGrid`:
        Used internally to represent the parameter grids.
    """

    def __init__(self, param_grid: Mapping[str, Any]):
        if not isinstance(param_grid, Mapping):
            if isinstance(param_grid, (list, Iterator)):
                raise TypeError("A sequence of grids (Iterable[Mapping[str, Any]) is not supported by this "
                                f"ParameterConfig ({param_grid}). Please use a "
                                "`timeeval.search.IndependentParameterGrid` for this!")
            else:
                raise TypeError(f"Parameter grid is not a dict ({param_grid})")
        self._param_grid = ParameterGrid(param_grid)

    @property
    def param_grid(self) -> ParameterGrid:
        return self._param_grid


class IndependentParameterGrid(ParameterConfig):
    """Grid of parameters with a discrete number of values for each.

    The parameters in the dict are considered independent and explored one after the other (no cartesian product).
    Uses the sklearn.model_selection.ParameterGrid internally.

    Parameters
    ----------
    param_grid : dict of str to sequence, or sequence of such
        The parameter grid to explore, as either
        - a dictionary mapping parameters to sequences of allowed values, or
        - a sequence of dicts signifying a sequence of grids to search.
        An empty dict signifies default parameters.

    default_params : dict of str to any values
        Default values for the parameters that are not in the current parameter grid.

    Examples
    --------
    >>> from timeeval.params import IndependentParameterGrid
    >>> params = {"a": [1, 2], "b": [True, False]}
    >>> default_params = {"a": 1, "b": True, "c": "auto"}
    >>> list(IndependentParameterGrid(params)) == ([
    ...     {"a": 1, "b": True, "c": "auto"},
    ...     {"a": 2, "b": True, "c": "auto"},
    ...     {"a": 1, "b": True, "c": "auto"},
    ...     {"a": 1, "b": False, "c": "auto"}
    ... ])
    True

    See also
    --------
    :class:`sklearn.model_selection.ParameterGrid`:
        Used internally to represent the parameter grids.
    """

    def __init__(self, param_grid: Mapping[str, Any], default_params: Optional[Mapping[str, Any]] = None):
        if not default_params:
            default_params = {}
        if not isinstance(param_grid, Mapping):
            raise TypeError(f"Parameter grid is not a dict ({param_grid})")
        if not isinstance(default_params, Mapping):
            raise TypeError(f"Default parameters is not a dict ({default_params})")

        self.default_params = {}
        for k, v in default_params.items():
            if isinstance(v, list):
                raise TypeError(f"Default parameters contain a list of values ({k}: {v}). Only fixed values are "
                                "allowed for defaults!")
            self.default_params[k] = [v]

        grids = []
        for param, values in param_grid.items():
            if not isinstance(values, list):
                values = [values]
            for v in values:
                grid = dict(self.default_params)
                grid[param] = [v]
                grids.append(grid)
        if len(grids) == 0:
            grids.append(self.default_params)
        self._param_grid = ParameterGrid(grids)

    @property
    def param_grid(self) -> ParameterGrid:
        return self._param_grid


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
    >>> from timeeval.params.search import FixedParameters
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
        self._param_grid = ParameterGrid({k: [v] for k, v in params.items()})

    @property
    def param_grid(self) -> ParameterGrid:
        return self._param_grid
