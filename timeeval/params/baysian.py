from typing import Iterator, Any, Mapping, Optional

from sklearn.model_selection import ParameterGrid

from .base import ParameterConfig


class OptunaParameterSearch(ParameterConfig):
    """Performs Bayesian optimization using Optuna library.

    .. warning::
        Please install the `optuna` package to use this class.
        We also recommend to install the `optuna-dashboard` package to visualize the optimization process.

    See also
    --------
    `https://optuna.readthedocs.io/en/stable/index.html`_:
        Optuna documentation.
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
