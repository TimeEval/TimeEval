from __future__ import annotations

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Optional, List, Tuple, Callable, Any, Dict

import docker
import optuna.storages
from docker.errors import DockerException, NotFound
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock, RDBStorage

from timeeval.integration import TimeEvalModule


if TYPE_CHECKING:
    from .config import OptunaConfiguration
    from ...timeeval import TimeEval
    from distributed import Scheduler
    from optuna.study import StudySummary


POSTGRESQL_IMAGE_NAME = "postgres:latest"
OPTUNA_DASHBOARD_IMAGE_NAME = "ghcr.io/optuna/optuna-dashboard:v0.8.1"
DB_CONTAINER_NAME = "timeeval-optuna-db"
DB_MAX_CONNECTIONS = 1000
DB_STARTUP_DELAY = 5  # in seconds
DASHBOARD_CONTAINER_NAME = "timeeval-optuna-dashboard"
log = logging.getLogger("OptunaModule")


# use an async function to not block the scheduler while waiting for the database to start up
async def _start_postgres_container(scheduler: Optional[Scheduler] = None, password: str = "postgres",
                                    port: int = 5432) -> None:
    import docker
    client = docker.from_env()
    log.debug(f"Starting postgres container on port {port}")
    client.containers.run(
        POSTGRESQL_IMAGE_NAME,
        f"-c max_connections={DB_MAX_CONNECTIONS}",
        name=DB_CONTAINER_NAME,
        environment={
            "POSTGRES_PASSWORD": password,
        },
        ports={"5432/tcp": port},
        detach=True,
    )
    log.debug(f"Waiting {DB_STARTUP_DELAY} seconds for database to start up")
    await asyncio.sleep(DB_STARTUP_DELAY)


def _start_dashboard_container(scheduler: Optional[Scheduler] = None,
                               storage: str = "postgresql://postgres:postgres@localhost:5432/postgres") -> None:
    import docker
    client = docker.from_env()
    log.debug("Starting dashboard container")
    client.containers.run(
        OPTUNA_DASHBOARD_IMAGE_NAME,
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
    """This module is automatically loaded when at least one algorithm uses
    :class:`timeeval.params.BayesianParameterSearch` as parameter config.

    TimeEval provides the option to use an automatically managed PostgreSQL database as the storage backend for the
    Optuna studies. The database is started as an additional Docker container either on the local machine or on the
    scheduler node in distributed execution mode. The database is automatically stopped when TimeEval is finished. The
    database storage backend allows you to monitor the studies using the Optuna dashboard (that can also be started
    automatically using another Docker container) and the distributed execution of the studies. This is the default
    behavior if no storage backend is specified in the configuration.

    Parameters
    ----------
    config : OptunaConfiguration
        The configuration for the Optuna module.
    """
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

        self.storage_url: Optional[str] = None

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
            storage_url = f"postgresql://postgres:{password}@{host}:{port}/postgres"
            self.config.default_storage = lambda: RDBStorage(
                storage_url,
                engine_kwargs={"pool_size": 1, "max_overflow": 2, "pool_timeout": 60}
            )
            log.debug(f"starting managed postgresql storage backend at ({storage_url})...")
            self.storage_url = storage_url

            tasks.append((_start_postgres_container, [], {"password": password, "port": port}))

        elif isinstance(self.config.default_storage, str) and self.config.default_storage == "journal-file":
            journal_file_path = str(timeeval.results_path / "optuna-journal.log")
            self.config.default_storage = lambda: JournalStorage(
                JournalFileStorage(journal_file_path, lock_obj=JournalFileOpenLock(journal_file_path))
            )

        if self.config.dashboard:
            if self.storage_url is None and not isinstance(self.config.default_storage, str):
                # user provided a custom storage backend, try to get the URL:
                from sqlalchemy.exc import OperationalError
                try:
                    storage = optuna.storages.get_storage(self.config.default_storage())
                    if hasattr(storage, "url"):
                        self.storage_url = storage.url  # type: ignore
                    else:
                        log.warning(f"Could not find dashboard connection URL for storage {self.config.default_storage}, "
                                    "not starting dashboard!")
                except OperationalError as e:
                    log.warning("Could not find connection URL for storage, not starting dashboard!", exc_info=e)

            if self.storage_url is not None:
                storage_host = timeeval.remote.config.scheduler_host if timeeval.distributed else "localhost"
                dashboard_url = f"http://{storage_host}:8080"
                log.debug(f"starting managed Optuna dashboard ({dashboard_url})...")
                print(f"\n\tOpen Optuna dashboard at {dashboard_url}\n")
                tasks.append((_start_dashboard_container, [], {"storage": self.storage_url}))

        if timeeval.distributed:
            log.debug(f"distributed mode: starting {len(tasks)} containers on scheduler")
            timeeval.remote.run_on_scheduler(tasks, msg="Starting Optuna containers on scheduler")
        else:
            for task, args, kwargs in tasks:
                task(*args, **kwargs)

        # update optuna study configurations:
        log.debug("updating optuna study configurations ...")
        for algo in timeeval.exps.algorithms:
            if hasattr(algo.param_config, "update_config"):
                algo.param_config.update_config(self.config)  # type: ignore
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
        """Load all studies from the default storage. This does not include studies, which were stored in a different
        storage backend (i.a. where the storage backend was changed using the
        :class:`timeeval.integration.optuna.OptunaStudyConfiguration`).

        Returns
        -------
        study_summaries : List[StudySummary]
            A list of study summaries.

        See Also
        --------
        :func:`optuna.study.get_all_study_summaries` : Optuna function which is used to load the studies.
        :class:`optuna.study.StudySummary` : Optuna class which is used to represent the study summaries.
        """
        return optuna.get_all_study_summaries(
            self.config.default_storage if isinstance(self.config.default_storage, str) else self.config.default_storage()
        )
