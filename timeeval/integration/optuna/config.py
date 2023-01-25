from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional

from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage
from optuna.study import StudyDirection

from timeeval import Metric


@dataclass(init=True, repr=True)
class OptunaConfiguration:
    """Configuration options for the Optuna module, which is automatically loaded when at least one
    algorithm uses :class:`timeeval.params.BayesianParameterSearch` as parameter config.

    Parameters
    ----------
    default_storage : str or optuna.storages.BaseStorage
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

    See Also
    --------
    :meth:`optuna.create_study`:
        Used to create the Optuna study object; includes detailed explanation of the parameters.
    """

    default_storage: Union[str, BaseStorage]
    default_sampler: Optional[BaseSampler] = None
    default_pruner: Optional[BasePruner] = None
    continue_existing_studies: bool = False
    dashboard: bool = False
    remove_managed_containers: bool = False

    @staticmethod
    def default(distributed: bool) -> OptunaConfiguration:
        if distributed:
            return OptunaConfiguration(default_storage="postgresql")
        else:
            return OptunaConfiguration(default_storage="journal-file")


@dataclass(init=True, repr=True, frozen=True)
class OptunaStudyConfiguration:
    """Configuration for :class:`OptunaParameterSearch`.

    Parameters
    ----------
    n_trials : int
        Number of trials to perform.
    metric : Metric
        TimeEval metric to use as the studies objective function.
    storage : str or optuna.storages.BaseStorage, optional
        Storage to store the results of the study.
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
    storage: Optional[Union[str, BaseStorage]] = None
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
