from asyncio import Future, run_coroutine_threadsafe, get_event_loop
from typing import List, Callable, Optional, Tuple, Dict
from dask.distributed import Client, SSHCluster
import tqdm
import time


class Remote:
    def __init__(self, **kwargs):
        self.ssh_cluster_kwargs = kwargs or {}
        self.client: Optional[Client] = None
        self.cluster: Optional[SSHCluster] = None
        self.futures: List[Future] = []

    def _start_cluster(self):
        self.cluster = SSHCluster(**self.ssh_cluster_kwargs)
        self.client = Client(self.cluster.scheduler_address)

    def add_task(self, task: Callable, *args, config: Optional[dict] = None, **kwargs) -> Future:
        config = config or {}
        if not self.client:
            self._start_cluster()
        assert self.client is not None
        future = self.client.submit(task, *args, **config, **kwargs)
        self.futures.append(future)
        return future

    def run_on_all_hosts(self, tasks: List[Tuple[Callable, List, Dict]]):
        if not self.client:
            self._start_cluster()
        assert self.client is not None
        futures = []
        for task, args, kwargs in tasks:
            for host in self.ssh_cluster_kwargs.get("hosts"):
                futures.append(self.add_task(task, *args, config={"workers": [host]}, **kwargs))
        self.client.gather(futures)

    def prune(self):
        if self.cluster:
            self.client.containers.prune()

    def fetch_results(self):
        n_experiments = len(self.futures)
        coroutine_future = run_coroutine_threadsafe(self.client.gather(self.futures, asynchronous=True), get_event_loop())

        progress_bar = tqdm.trange(n_experiments, desc="Evaluating distributedly", position=0)  # todo add repetitions

        while not coroutine_future.done():
            n_done = sum([f.done() for f in self.futures])
            progress_bar.update(n_done - progress_bar.n)
            if progress_bar.n == n_experiments:
                break
            time.sleep(0.5)
        progress_bar.update(n_experiments - progress_bar.n)

        progress_bar.close()

    def close(self):
        self.client.shutdown()
        self.client.close()
        self.cluster.close()
