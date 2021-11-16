import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict

from .resource_constraints import ResourceConstraints


DEFAULT_SCHEDULER_HOST = "localhost"
DEFAULT_DASK_PORT = 8786
DEFAULT_DASK_LOG_FILENAME = "dask.log"


@dataclass
class RemoteConfiguration:
    scheduler_host: str = DEFAULT_SCHEDULER_HOST
    scheduler_port: int = DEFAULT_DASK_PORT
    worker_hosts: List[str] = field(default_factory=lambda: [])
    remote_python: str = sys.executable
    kwargs_overwrites: dict = field(default_factory=lambda: {})
    dask_logging_file_level: str = "INFO"
    dask_logging_console_level: str = "INFO"
    dask_logging_filename: str = DEFAULT_DASK_LOG_FILENAME

    def update_logging_path(self, results_path: Path) -> None:
        name = self.dask_logging_filename
        path = results_path / name
        self.dask_logging_filename = str(path.resolve())

    def to_ssh_cluster_kwargs(self, limits: ResourceConstraints):
        """
        Creates the kwargs for the Dask SSHCluster-constructor:
        https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster
        """
        mem_limit = limits.task_memory_limit or "auto"
        config = {
            "hosts": [self.scheduler_host] + self.worker_hosts,
            # "connect_options": {
            #     "port": self.scheduler_port
            # },
            # https://distributed.dask.org/en/latest/worker.html?highlight=worker_options#distributed.worker.Worker
            "worker_options": {
                "nprocs": limits.tasks_per_host,
                "nthreads": 1,
                "memory_limit": mem_limit,
            },
            # defaults are fine: https://distributed.dask.org/en/latest/scheduling-state.html?highlight=dask.distributed.Scheduler#distributed.scheduler.Scheduler
            # "scheduler_options": {},
            # "worker_module": "distributed.cli.dask_worker",  # default
            "remote_python": self.remote_python
        }
        config.update(self.kwargs_overwrites)
        return config

    def get_remote_logging_config(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "incremental": False,
            "formatters": {
                "brief": {
                    "format": "%(name)s - %(levelname)s - %(message)s"
                },
                "verbose-file": {
                    "format": "%(asctime)s - %(levelname)s - %(process)d %(name)s - %(message)s"
                }
            },
            "handlers": {
                "stdout": {
                    "level": self.dask_logging_console_level.upper(),
                    "formatter": "brief",
                    "class": "logging.StreamHandler"
                },
                "log_file": {
                    "level": self.dask_logging_file_level.upper(),
                    "formatter": "verbose-file",
                    "filename": self.dask_logging_filename,
                    "class": "logging.FileHandler",
                    "mode": "a"
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["stdout", "log_file"]
            }
        }
