import logging
import sys
import time
from asyncio import Future, run_coroutine_threadsafe, get_event_loop
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Tuple, Dict, Iterator

import tqdm
from dask.distributed import Client, SSHCluster

DEFAULT_SCHEDULER_HOST = "localhost"
DEFAULT_DASK_PORT = 8786


@dataclass
class RemoteConfiguration:
    scheduler_host: str = DEFAULT_SCHEDULER_HOST
    scheduler_port: int = DEFAULT_DASK_PORT
    worker_hosts: List[str] = field(default_factory=lambda: [])
    remote_python: str = sys.executable
    kwargs_overwrites: dict = field(default_factory=lambda: {})

    def to_ssh_cluster_kwargs(self):
        """
        Creates the kwargs for the Dask SSHCluster-constructor:
        https://docs.dask.org/en/latest/setup/ssh.html#distributed.deploy.ssh.SSHCluster
        """
        config = {
            "hosts": [self.scheduler_host] + self.worker_hosts,
            "connect_options": {
                "port": self.scheduler_port
            },
            # https://distributed.dask.org/en/latest/worker.html?highlight=worker_options#distributed.worker.Worker
            "worker_options": {
                "ncores": 1,
                "nthreads": 1
            },
            # defaults are fine: https://distributed.dask.org/en/latest/scheduling-state.html?highlight=dask.distributed.Scheduler#distributed.scheduler.Scheduler
            "scheduler_options": {},
            "worker_module": "distributed.cli.dask_worker",  # default
            "remote_python": self.remote_python
        }
        config.update(self.kwargs_overwrites)
        return config


class Remote:
    def __init__(self, disable_progress_bar: bool = False, remote_config: RemoteConfiguration = RemoteConfiguration()):
        self.logger = logging.getLogger("Remote")
        self.config = remote_config
        self.futures: List[Future] = []
        self.cluster = self.start_or_restart_cluster()
        self.client = Client(self.cluster.scheduler_address)
        self.disable_progress_bar = disable_progress_bar

    def start_or_restart_cluster(self, n=0) -> SSHCluster:
        if n >= 5:
            raise RuntimeError("Could not start an SSHCluster because there is already one running, "
                               "that cannot be stopped!")
        try:
            return SSHCluster(**self.config.to_ssh_cluster_kwargs())
        except Exception as e:
            if "Worker failed to start" in str(e):
                scheduler_host = self.config.scheduler_host
                port = self.config.scheduler_port
                scheduler_address = f"{scheduler_host}:{port}"

                self.logger.warning(f"Failed to start cluster, because address already in use! "
                                    f"Trying to restart cluster at {scheduler_address}!")

                with Client(scheduler_address) as client:
                    client.shutdown()

                return self.start_or_restart_cluster(n+1)
            raise e

    def add_task(self, task: Callable, *args, config: Optional[dict] = None, **kwargs) -> Future:
        config = config or {}
        future = self.client.submit(task, *args, **config, **kwargs)
        self.futures.append(future)
        return future

    def run_on_all_hosts(self, tasks: List[Tuple[Callable, List, Dict]]):
        for task, args, kwargs in tasks:
            self.client.run(task, *args, **kwargs)

    def fetch_results(self):
        n_experiments = len(self.futures)
        coroutine_future = run_coroutine_threadsafe(self.client.gather(self.futures, asynchronous=True), get_event_loop())
        progress_bar = tqdm.trange(n_experiments, desc="Evaluating distributedly", position=0, disable=self.disable_progress_bar)

        while not coroutine_future.done():
            n_done = sum([f.done() for f in self.futures])
            progress_bar.update(n_done - progress_bar.n)
            if progress_bar.n == n_experiments:
                break
            time.sleep(0.5)

        progress_bar.update(n_experiments - progress_bar.n)
        progress_bar.close()

    def close(self):
        self.cluster.close()
        self.client.shutdown()
        self.client.close()
