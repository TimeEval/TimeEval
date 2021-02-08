from asyncio import Future, run_coroutine_threadsafe, get_event_loop
from typing import List, Callable, Optional, Tuple, Dict
from dask.distributed import Client, SSHCluster
import tqdm
import time
import logging


DEFAULT_DASK_PORT = 8786


class Remote:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger("Remote")
        self.ssh_cluster_kwargs = kwargs or {}
        self.futures: List[Future] = []
        self.cluster = self.start_or_restart_cluster()
        self.client = Client(self.cluster.scheduler_address)

    def start_or_restart_cluster(self, n=0) -> SSHCluster:
        if n >= 5:
            raise RuntimeError("Could not start an SSHCluster because there is already one running, "
                               "that cannot be stopped!")
        try:
            return SSHCluster(**self.ssh_cluster_kwargs)
        except Exception as e:
            if "Worker failed to start" in str(e):
                port = self.ssh_cluster_kwargs.get("connect_options", {}).get("port", DEFAULT_DASK_PORT)
                scheduler_host = self.ssh_cluster_kwargs.get("hosts")[0]
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

        progress_bar = tqdm.trange(n_experiments, desc="Evaluating distributedly", position=0)

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
