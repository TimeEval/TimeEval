from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.storages import BaseStorage, RDBStorage
from optuna.trial import TrialState

from timeeval.params import ParameterConfig, Params


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    from typing import Iterator, Any, Mapping, Dict, Optional, ItemsView, Union
    from optuna import Study, Trial
    from optuna.distributions import BaseDistribution
    from timeeval import Algorithm
    from timeeval.datasets import Dataset
    from .config import OptunaConfiguration, OptunaStudyConfiguration


@dataclass(init=False, repr=True)
class OptunaLazyParams(Params):
    study_name: str
    index: int
    distributions: Dict[str, BaseDistribution]
    config: OptunaStudyConfiguration

    def __init__(self,
                 study_name: str,
                 index: int,
                 distributions: Mapping[str, BaseDistribution],
                 config: OptunaStudyConfiguration,
                 ):
        super().__init__()
        self.study_name = study_name
        self.index = index
        self.distributions = dict(distributions)
        self.config = config
        self._study: Optional[Study] = None
        self._trial: Optional[Trial] = None

    def __len__(self) -> int:
        return len(self.distributions)

    def __iter__(self) -> Iterator[str]:
        return iter(self.distributions)

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

    def items(self) -> ItemsView[str, Any]:
        return dict((k, self.uid()) for k in self.distributions).items()

    def materialize(self) -> Params:
        # only materialize once:
        if self._study is None:
            if self.config.storage is None or isinstance(self.config.storage, str):
                storage: Union[None, str, BaseStorage] = self.config.storage
            else:
                storage = self.config.storage()
            self._study = optuna.load_study(
                study_name=self.study_name,
                storage=storage,
                sampler=self.config.sampler,
                pruner=self.config.pruner,
            )
            # we need to manually reseed the sampler because we are not in an optimize()-call:
            self._study.sampler.reseed_rng()

        if self._trial is None:
            self._trial = self._study.ask(self.distributions)
            self._trial.set_user_attr("uid", self.uid())
            self._trial.set_user_attr("node", socket.gethostname())
        return self

    def assess(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        t = self.trial()
        score = self.config.metric(y_true, y_score)
        self.study().tell(t, score)
        self.close()
        return score

    def fail(self) -> None:
        self.study().tell(self.trial(), state=TrialState.FAIL)
        self.close()

    def uid(self) -> str:
        return self.build_uid(self.study_name, self.index)

    def to_dict(self) -> Dict[str, Any]:
        params: Dict[str, Any] = self.trial().params
        return params

    def close(self) -> None:
        if self._trial is not None:
            del self._trial
        if self._study is not None:
            storage = self._study._storage
            if isinstance(storage, RDBStorage):
                storage.remove_session()
            del self._study

    @staticmethod
    def build_uid(study_name: str, i: int) -> str:
        return f"{study_name}-{i}"


class OptunaParameterSearch(ParameterConfig):
    """Implementation of the Bayesian optimization using Optuna library."""

    def __init__(self, config: OptunaStudyConfiguration,
                 params: Mapping[str, BaseDistribution],
                 include_default_params: bool = False):
        self._config = config
        self._distributions = params
        self._include_default_params = include_default_params

    def iter(self, algorithm: Algorithm, dataset: Dataset) -> Iterator[Params]:
        # create the study and enforce a common name for all trials of the study (this will create the study in the
        # storage backend so that it can be accessed by all workers):
        if self._config.storage is None or isinstance(self._config.storage, str):
            storage: Union[None, str, BaseStorage] = self._config.storage
        else:
            storage = self._config.storage()
        study = optuna.create_study(
            study_name=f"{algorithm.name}-{dataset.name}",
            storage=storage,
            sampler=self._config.sampler,
            pruner=self._config.pruner,
            direction=self._config.direction,
            load_if_exists=self._config.continue_existing_study,
        )
        study.set_user_attr("algorithm", algorithm.name)
        study.set_user_attr("dataset", dataset.name)
        study.set_user_attr("metric", self._config.metric.name)
        study.set_user_attr("direction", str(self._config.direction).lower())
        study.set_user_attr("includes_default_params", self._include_default_params)
        study_name = study.study_name

        if self._include_default_params:
            try:
                study.enqueue_trial(self._default_params(algorithm))
                yield OptunaLazyParams(study_name, 0, self._distributions, self._config)
            except ValueError as e:
                from .module import log
                log.warning(f"Could not create default parameters for {algorithm.name}, skipping!", exc_info=e)

        # cleanup (each trial manages their own DB connection)
        del study
        if isinstance(storage, RDBStorage):
            storage.remove_session()

        for i in range(self._config.n_trials):
            yield OptunaLazyParams(study_name, i + self._include_default_params, self._distributions, self._config)

    def __len__(self) -> int:
        return self._config.n_trials + int(self._include_default_params)

    def update_config(self, global_config: OptunaConfiguration) -> None:
        self._config = self._config.update_unset_options(global_config)

    def _default_params(self, algorithm: Algorithm) -> Dict[str, Any]:
        params = {}
        for param_name in self._distributions:
            if param_name not in algorithm.param_schema:
                continue
            schema = algorithm.param_schema[param_name]
            value = schema["defaultValue"]
            tpe = schema["type"].lower()

            if value is None or "enum" in tpe:
                params[param_name] = value
            elif tpe == "int":
                params[param_name] = int(value)
            elif tpe == "float":
                params[param_name] = float(value)
            elif tpe == "boolean":
                params[param_name] = bool(value)
            elif tpe == "str":
                params[param_name] = str(value)
            else:
                raise ValueError(f"Unsupported parameter type {tpe} for param {param_name} in {algorithm.name}!")

        if len(params) == 0:
            raise ValueError(f"No default parameters found for {algorithm.name}!")

        return params
