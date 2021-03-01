import sys
from dataclasses import dataclass, field
from typing import List

DEFAULT_SCHEDULER_HOST = "localhost"
DEFAULT_DASK_PORT = 8786


@dataclass
class RemoteConfiguration:
    scheduler_host: str = DEFAULT_SCHEDULER_HOST
    scheduler_port: int = DEFAULT_DASK_PORT
    worker_hosts: List[str] = field(default_factory=lambda: [])
    remote_python: str = sys.executable
    kwargs_overwrites: dict = field(default_factory=lambda: {})

    def to_ssh_cluster_kwargs(self, tasks_per_host: int):
        """
        Creates the kwargs for the Dask SSHCluster-constructor:
        https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster
        """
        config = {
            "hosts": [self.scheduler_host] + self.worker_hosts,
            # "connect_options": {
            #     "port": self.scheduler_port
            # },
            # https://distributed.dask.org/en/latest/worker.html?highlight=worker_options#distributed.worker.Worker
            # "worker_options": {
            #     "ncores": tasks_per_host,
            #     "nthreads": 1
            # },
            # defaults are fine: https://distributed.dask.org/en/latest/scheduling-state.html?highlight=dask.distributed.Scheduler#distributed.scheduler.Scheduler
            # "scheduler_options": {},
            # "worker_module": "distributed.cli.dask_worker",  # default
            "remote_python": self.remote_python
        }
        config.update(self.kwargs_overwrites)
        return config
