from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Iterator, Any, Mapping, Dict, Union, Optional

import numpy as np
import optuna
from optuna import Study, Trial
from optuna.distributions import BaseDistribution
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection
from optuna.trial import TrialState

from .base import ParameterConfig
from .params import Params
from ..metrics import Metric
from ..datasets import Dataset


@dataclass(init=True, repr=True, frozen=True)
class OptunaConfiguration:
    """Configuration for :class:`OptunaParameterSearch`.

    Parameters
    ----------
    n_trials : int
        Number of trials to perform.
    metric : Metric
        TimeEval metric to use as the studies objective function.
    storage : str or optuna.storages.BaseStorage
        Storage to store the results of the study. This is required for TimeEval.
    sampler : optuna.samplers.BaseSampler, optional
        Sampler to use for the study. If not provided, the default sampler is used.
    pruner : optuna.pruners.BasePruner, optional
        Pruner to use for the study. If not provided, the default pruner is used.
    direction : str or optuna.study.StudyDirection, optional
        Direction of the optimization (minimize or maximize). If not provided, the default direction is used.
    continue_existing : bool, optional
        If True, continue a study with the given name if it already exists in the storage backend. If False, raise an
        error if a study with the same name already exists.

    See Also
    --------
    :meth:`optuna.create_study`:
        Used to create the Optuna study object; includes detailed explanation of the parameters.
    """

    n_trials: int
    metric: Metric
    storage: Union[str, BaseStorage]
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    direction: Optional[Union[str, StudyDirection]] = "maximize"
    continue_existing_study: bool = False


class OptunaLazyParams(Params):
    def __init__(self,
                 study_name: str,
                 uid: int,
                 distributions: Mapping[str, BaseDistribution],
                 config: OptunaConfiguration,
                 ):
        super().__init__()
        self._study_name = study_name
        self._uid = uid
        self._distributions = dict(distributions)
        self._config = config
        self._study: Optional[Study] = None
        self._trial: Optional[Trial] = None

    def __len__(self) -> int:
        return len(self._distributions)

    def __iter__(self) -> Iterator[str]:
        return iter(self._distributions)

    def __getitem__(self, param_name: str) -> Any:
        return self.trial().params[param_name]

    def trial(self) -> Trial:
        if self._trial is None:
            raise ValueError("The parameters have not yet been materialized!")
        return self._trial

    def study(self) -> Study:
        if self._study is None:
            raise ValueError("The parameters have not yet been materialized!")
        return self._study

    def items(self) -> Iterator[Any]:
        return ((k, self._uid) for k in self._distributions)

    def materialize(self) -> Params:
        # only materialize once:
        if self._study is None:
            self._study = optuna.load_study(
                study_name=self._study_name,
                storage=self._config.storage,
                sampler=self._config.sampler,
                pruner=self._config.pruner,
            )
        if self._trial is None:
            self._trial = self._study.ask(self._distributions)
            self._trial.set_user_attr("uid", self.uid())
            self._trial.set_user_attr("node", socket.gethostname())
        return self

    def assess(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        t = self.trial()
        score = self._config.metric(y_true, y_score)
        self.study().tell(t, score)
        return score

    def fail(self) -> None:
        self.study().tell(self.trial(), state=TrialState.FAIL)

    def uid(self) -> str:
        return self.build_uid(self._study_name, self._uid)

    def to_dict(self) -> Dict[str, Any]:
        return self.trial().params

    @staticmethod
    def build_uid(study_name: str, i: int) -> str:
        return f"{study_name}-{i}"


class OptunaParameterSearch(ParameterConfig):
    """Performs Bayesian optimization using Optuna library.

    .. warning::
        Please install the `optuna` package to use this class.
        We also recommend to install the `optuna-dashboard` package to visualize the optimization process.
        If you use the recommended PostgreSQL storage backend, you also need to install the `psycopg2` package.

    See also
    --------
    `https://optuna.readthedocs.io/en/stable/index.html`_:
        Optuna documentation.
    """

    def __init__(self, config: OptunaConfiguration, params: Mapping[str, BaseDistribution]):
        self._config = config
        self._distributions = params

    def iter(self, algorithm: "Algorithm", dataset: Dataset) -> Iterator[Params]:
        # create the study and enforce a common name for all trials of the study (this will create the study in the
        # storage backend so that it can be accessed by all workers):
        study = optuna.create_study(
            study_name=f"{algorithm.name}-{dataset.name}",
            storage=self._config.storage,
            sampler=self._config.sampler,
            pruner=self._config.pruner,
            direction=self._config.direction,
            load_if_exists=self._config.continue_existing_study,
        )
        study.set_user_attr("algorithm", algorithm.name)
        study.set_user_attr("dataset", dataset.name)
        study.set_user_attr("metric", self._config.metric.name)
        study.set_user_attr("direction", str(self._config.direction).lower())
        study_name = study.study_name
        del study
        for i in range(self._config.n_trials):
            yield OptunaLazyParams(study_name, i, self._distributions, self._config)

    def __len__(self) -> int:
        return self._config.n_trials

# docker run --name postgres -e POSTGRES_PASSWORD=hairy_bumblebee -p 5432:5432 -d postgres
# optuna-dashboard postgresql://postgres:hairy_bumblebee@localhost:5432/postgres
