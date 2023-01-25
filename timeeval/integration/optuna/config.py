from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional

from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.storages import BaseStorage


@dataclass(init=True, repr=True)
class OptunaConfiguration:
    """Configuration options for the Optuna module, which is automatically loaded when at least one
    algorithm uses :class:`timeeval.params.baysian.OptunaSearchParameters` as parameter config.

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
