from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Optional, List, Tuple, Callable, Any, Dict

import docker
import optuna.storages
from docker.errors import DockerException, NotFound
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock

from .. import TimeEvalModule
from ...params.baysian import OptunaParameterSearch


if TYPE_CHECKING:
    from .config import OptunaConfiguration
    from ...timeeval import TimeEval
    from distributed import Scheduler
    from optuna.study import StudySummary


DB_CONTAINER_NAME = "timeeval-optuna-db"
DASHBOARD_CONTAINER_NAME = "timeeval-optuna-dashboard"
STARTUP_DELAY = 5
log = logging.getLogger("OptunaModule")


# use an async function to not block the scheduler while waiting for the database to start up
async def start_postgres_container(scheduler: Optional[Scheduler] = None, password: str = "postgres",
                                   port: int = 5432) -> None:
    import docker
    client = docker.from_env()
    log.debug(f"Starting postgres container on port {port}")
    client.containers.run(
        "postgres:latest",
        name=DB_CONTAINER_NAME,
        environment={
            "POSTGRES_PASSWORD": password,
        },
        ports={"5432/tcp": port},
        detach=True,
    )
    log.debug(f"Waiting {STARTUP_DELAY} seconds for database to start up")
    await asyncio.sleep(STARTUP_DELAY)


def _start_dashboard_container(scheduler: Optional[Scheduler] = None,
                               storage: str = "postgresql://postgres:postgres@localhost:5432/postgres") -> None:
    import docker
    client = docker.from_env()
    log.debug("Starting dashboard container")
    client.containers.run(
        "ghcr.io/optuna/optuna-dashboard:latest",
        storage,
        name=DASHBOARD_CONTAINER_NAME,
        network_mode="host",
        # ports={"8080/tcp": 8080},
        # extra_hosts={"host.docker.internal": "host-gateway"},
        detach=True,
    )


def _stop_containers(scheduler: Optional[Scheduler] = None, remove: bool = False) -> None:
    import docker
    client = docker.from_env()
    try:
        c = client.containers.get(DASHBOARD_CONTAINER_NAME)
        log.debug("Stopping dashboard container")
        c.stop()
        if remove:
            c.remove(v=True, force=True)
    except NotFound:
        pass
    try:
        c = client.containers.get(DB_CONTAINER_NAME)
        log.debug("Stopping database container")
        c.stop()
        if remove:
            c.remove(v=True, force=True)
    except NotFound:
        pass


class OptunaModule(TimeEvalModule):
    def __init__(self, config: OptunaConfiguration):
        self.config = config
        # check configuration:
        check_docker = ""
        if self.config.dashboard:
            check_docker = "start the Optuna dashboard"
        if self.config.default_storage is None:
            raise ValueError("No default storage specified!")
        elif isinstance(self.config.default_storage, str) and self.config.default_storage == "postgresql":
            and_required = " and " if check_docker else ""
            check_docker += and_required + f"start the Optuna storage {self.config.default_storage}"
        self._check_docker_available(check_docker)

        self.dashboard_storage_url: Optional[str] = None

    @staticmethod
    def _check_docker_available(reason: str) -> None:
        if reason:
            try:
                docker.from_env()
            except DockerException as e:
                raise ValueError(f"No docker client found, but docker is required to {reason}!") from e

    def prepare(self, timeeval: TimeEval) -> None:
        log.info("Optuna module: preparing ...")
        tasks: List[Tuple[Callable, List[Any], Dict[str, Any]]] = []

        if isinstance(self.config.default_storage, str) and self.config.default_storage == "postgresql":
            host = timeeval.remote.config.scheduler_host if timeeval.distributed else socket.gethostname()
            port = 5432
            password = "hairy_bumblebee"
            # set actual connection string:
            self.config.default_storage = f"postgresql://postgres:{password}@{host}:{port}/postgres"
            log.debug(f"starting managed postgresql storage backend ({self.config.default_storage})...")

            tasks.append((start_postgres_container, [], {"password": password, "port": port}))

        elif isinstance(self.config.default_storage, str) and self.config.default_storage == "journal-file":
            journal_file_path = str(timeeval.results_path / "optuna-journal.log")
            self.config.default_storage = JournalStorage(
                JournalFileStorage(journal_file_path, lock_obj=JournalFileOpenLock(journal_file_path))
            )

        if self.config.dashboard:
            if isinstance(self.config.default_storage, str):
                self.dashboard_storage_url = self.config.default_storage
            else:
                storage = optuna.storages.get_storage(self.config.default_storage)
                if hasattr(storage, "url"):
                    self.dashboard_storage_url = storage.url  # type: ignore
                else:
                    self.dashboard_storage_url = None
                    log.warning(f"Could not find dashboard connection URL for storage {self.config.default_storage}, "
                                "not starting dashboard!")

            if self.dashboard_storage_url is not None:
                storage_host = timeeval.remote.config.scheduler_host if timeeval.distributed else "localhost"
                dashboard_url = f"http://{storage_host}:8080"
                log.debug(f"starting managed Optuna dashboard ({dashboard_url})...")
                print(f"\n\tOpen Optuna dashboard at {dashboard_url}\n")
                tasks.append((_start_dashboard_container, [], {"storage": self.dashboard_storage_url}))

        if timeeval.distributed:
            log.debug(f"distributed mode: starting {len(tasks)} containers on scheduler")
            timeeval.remote.run_on_scheduler(tasks, msg="Starting Optuna containers on scheduler")
        else:
            for task, args, kwargs in tasks:
                task(*args, **kwargs)

        # update optuna study configurations:
        log.debug("updating optuna study configurations ...")
        for algo in timeeval.exps.algorithms:
            if isinstance(algo.param_config, OptunaParameterSearch):
                algo.param_config.update_config(self.config)
        log.info("Optuna module: preparing done.")

    def finalize(self, timeeval: TimeEval) -> None:
        log.info("Optuna module: finalizing ...")
        if timeeval.distributed:
            log.debug("distributed mode: stopping containers on scheduler")
            timeeval.remote.run_on_scheduler(
                [(_stop_containers, [], {"remove": self.config.remove_managed_containers})],
                msg="Stopping Optuna containers on scheduler"
            )
        else:
            _stop_containers(remove=self.config.remove_managed_containers)
        log.info("Optuna module: finalizing done.")

    def load_studies(self) -> List[StudySummary]:
        """Load all studies from the default storage. This does not include studies which were stored in a different
        storage backend (i.a. where the storage backend was changed using the
        :class:`timeeval.params.baysian.OptunaStudyConfiguration`).

        Returns
        -------
        study_summaries : List[StudySummary]
            A list of study summaries.

        See Also
        --------
        :func:`optuna.study.get_all_study_summaries` : Optuna function which is used to load the studies.
        :class:`optuna.study.StudySummary` : Optuna class which is used to represent the study summaries.
        """
        return optuna.get_all_study_summaries(self.config.default_storage)
