from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ParameterConfig


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from typing import Mapping, Iterator
    from .params import Params
    from ..algorithm import Algorithm
    from ..datasets import Dataset
    from ..integration.optuna import OptunaConfiguration, OptunaStudyConfiguration


class BayesianParameterSearch(ParameterConfig):
    """Performs Bayesian optimization using Optuna integration.

    .. warning::
        Please install the `Optuna <https://optuna.org>`_ package to use this class.
        If you use the recommended PostgreSQL storage backend, you also need to install the `psycopg2` or
        `psycopg2-binary` package:

        .. code-block:: bash

            pip install optuna>=3.1.0 psycopg2

    Parameters
    ----------
    config : OptunaStudyConfiguration
        Configuration for the Optuna study. Optional parameters are filled in with the default values from the glabal
        Optuna configuration.
    params : Mapping[str, BaseDistribution]
        Mapping from parameter names to the corresponding Optuna distributions.

    Examples
    --------
    >>> from timeeval.params import BayesianParameterSearch
    >>> from timeeval.metrics import RangePrAUC
    >>> from timeeval.integration.optuna import OptunaStudyConfiguration
    >>> from optuna.distributions import FloatDistribution, IntDistribution
    >>> config = OptunaStudyConfiguration(n_trials=10, metric=RangePrAUC())
    >>> distributions = {
    ...     "max_features": FloatDistribution(low=0.0, high=1.0, step=0.01),
    ...     "window_size": IntDistribution(low=5, high=1000, step=5),
    ... }
    >>> BayesianParameterSearch(config, distributions)
    <timeeval.params.bayesian.BayesianParameterSearch object at 0x7f9cdc8faf50>

    See also
    --------
    `<https://optuna.readthedocs.io>`_:
        Optuna documentation.
    :class:`timeeval.integration.optuna.OptunaModule`:
        Optuna integration TimeEval module.
    """

    def __init__(self, config: OptunaStudyConfiguration, params: Mapping[str, BaseDistribution]):
        from ..integration.optuna import OptunaParameterSearch
        self._impl = OptunaParameterSearch(config, params)

    def iter(self, algorithm: Algorithm, dataset: Dataset) -> Iterator[Params]:
        return self._impl.iter(algorithm, dataset)

    def __len__(self) -> int:
        return self._impl.__len__()

    def update_config(self, global_config: OptunaConfiguration) -> None:
        """Updates unset / default values in the study configuration with the global configuration values.

        Parameters
        ----------
        global_config : OptunaConfiguration
            Global Optuna configuration.
        """
        self._impl.update_config(global_config)
