from __future__ import annotations

import logging

import optuna.logging
from dataclasses import dataclass
from typing import Union, Optional, Any, Callable

from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from timeeval import Metric


@dataclass(init=True, repr=True)
class OptunaConfiguration:
    """Configuration options for the Optuna module. This includes default options for all Optuna studies.

    Parameters
    ----------
    default_storage : str or Lambda returning instance of optuna.storages.BaseStorage
        Storage to store and synchronize the results of the studies. Per default, TimeEval will use a journal file in
        local execution mode and a PostgreSQL database in distributed execution mode. The database is automatically
        started and stopped by TimeEval using the latest postgres-Docker image. Use ``"postgresql"`` to let TimeEval
        handle starting and stopping a PostgreSQL database using Docker. Use ``"journal-file"`` to let TimeEval
        create a local file as the storage backend. This only works in non-distributed mode.
    default_sampler : optuna.samplers.BaseSampler, optional
        Sampler to use for the study. If not provided, the default sampler is used.
    default_pruner : optuna.pruners.BasePruner, optional
        Pruner to use for the study. If not provided, the default pruner is used.
    continue_existing_studies : bool, optional
        If True, continue a study with the given name if it already exists in the storage backend. If False, raise an
        error if a study with the same name already exists.
    dashboard : bool, optional
        If True, start the Optuna dashboard (within its own Docker container) to monitor the studies. In distributed
        execution mode, the dashboard is started on the scheduler node.
    remove_managed_containers : bool, optional
        If True, remove the containers managed by TimeEval (e.g., the PostgreSQL database) when TimeEval is finished.
    use_default_logging : bool, optional
        If True, use the default logging configuration of the Optuna library. This will log the progress of the studies
        to `stderr`. If False, use the logging configuration of TimeEval and propagates the Optuna log messages.
    log_level : int or str, optional
        The log level to use for the Optuna logger. The default is ``info`` = :attr:`logging.INFO` = ``20``.

    See Also
    --------
    :func:`optuna.create_study`:
        Used to create the Optuna study object; includes detailed explanation of the parameters.
    :class:`timeeval.integration.optuna.OptunaModule`:
        Optuna integration module for TimeEval.
    """

    default_storage: Union[str, Callable[[], BaseStorage]]
    default_sampler: Optional[BaseSampler] = None
    default_pruner: Optional[BasePruner] = None
    continue_existing_studies: bool = False
    dashboard: bool = False
    remove_managed_containers: bool = False
    use_default_logging: bool = False
    log_level: Union[int, str] = logging.INFO

    def __post_init__(self) -> None:
        self._update_optuna_logging()

    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)
        if key == "log_level" or key == "use_default_logging":
            # update the Optuna logging configuration when changed
            self._update_optuna_logging()

    def _update_optuna_logging(self) -> None:
        if self.use_default_logging:
            optuna.logging.enable_default_handler()
            optuna.logging.disable_propagation()
        else:
            optuna.logging.disable_default_handler()
            optuna.logging.enable_propagation()
        switcher = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level = switcher[self.log_level] if isinstance(self.log_level, str) else self.log_level
        optuna.logging.set_verbosity(level)

    @staticmethod
    def default(distributed: bool) -> OptunaConfiguration:
        if distributed:
            return OptunaConfiguration(default_storage="postgresql")
        else:
            return OptunaConfiguration(default_storage="journal-file")


@dataclass(init=True, repr=True, frozen=True)
class OptunaStudyConfiguration:
    """Configuration for :class:`~timeeval.params.BayesianParameterSearch`.

    The parameters ``n_trials`` and ``metric`` are required. All other parameters are optional and will be filled with
    the default values from the global Optuna configuration if not provided.

    Parameters
    ----------
    n_trials : int
        Number of trials to perform.
    metric : Metric
        TimeEval metric to use as the studies objective function.
    storage : str or Lambda returning instance of optuna.storages.BaseStorage, optional
        Storage to store the results of the study.
    sampler : optuna.samplers.BaseSampler, optional
        Sampler to use for the study. If not provided, the default sampler is used.
    pruner : optuna.pruners.BasePruner, optional
        Pruner to use for the study. If not provided, the default pruner is used.
    direction : str or optuna.study.StudyDirection, optional
        Direction of the optimization (minimize or `maximize`). If ``None``, the Optuna default direction is used.
    continue_existing_study : bool, optional
        If True, continue a study with the given name if it already exists in the storage backend. If False, raise an
        error if a study with the same name already exists.

    See Also
    --------
    :func:`optuna.create_study`:
        Used to create the Optuna study object; includes detailed explanation of the parameters.
    :class:`timeeval.integration.optuna.OptunaModule`:
        Optuna integration module for TimeEval.
    """

    n_trials: int
    metric: Metric
    storage: Optional[Union[str, Callable[[], BaseStorage]]] = None
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
    direction: Optional[Union[str, StudyDirection]] = "maximize"
    continue_existing_study: bool = False

    def update_unset_options(self, global_config: OptunaConfiguration) -> OptunaStudyConfiguration:
        return OptunaStudyConfiguration(
            n_trials=self.n_trials,
            metric=self.metric,
            storage=self.storage or global_config.default_storage,
            sampler=self.sampler or global_config.default_sampler,
            pruner=self.pruner or global_config.default_pruner,
            direction=self.direction,
            continue_existing_study=self.continue_existing_study or global_config.continue_existing_studies,
        )

    def copy(self,
             n_trials: Optional[int] = None,
             metric: Optional[Metric] = None,
             storage: Optional[Union[str, Callable[[], BaseStorage]]] = None,
             sampler: Optional[BaseSampler] = None,
             pruner: Optional[BasePruner] = None,
             direction: Optional[Union[str, StudyDirection]] = None,
             continue_existing_study: Optional[bool] = None) -> OptunaStudyConfiguration:
        """Create a copy of this configuration with the given parameters replaced."""

        return OptunaStudyConfiguration(
            n_trials=n_trials or self.n_trials,
            metric=metric or self.metric,
            storage=storage or self.storage,
            sampler=sampler or self.sampler,
            pruner=pruner or self.pruner,
            direction=direction or self.direction,
            continue_existing_study=continue_existing_study or self.continue_existing_study,
        )
