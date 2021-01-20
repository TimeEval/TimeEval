from asyncio import Future
from typing import List, Callable, Optional
from dask.distributed import Client, SSHCluster


class Remote:
    def __init__(self, **kwargs):
        self.ssh_cluster_kwargs = kwargs or dict()
        self.client: Optional[Client] = None
        self.cluster: Optional[SSHCluster] = None
        self.futures: List[Future] = []

    def _start_cluster(self):
        self.cluster = SSHCluster(**self.ssh_cluster_kwargs)
        self.client = Client(self.cluster)

    def add_task(self, task: Callable, *args) -> Future:
        if self.client is None:
            self._start_cluster()
        assert self.client is not None
        future = self.client.submit(task, *args)
        self.futures.append(future)
        return future

    def fetch_results(self):
        self.client.gather(self.futures)

    def close(self):
        self.client.close()
        self.cluster.close()
