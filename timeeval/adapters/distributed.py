import numpy as np
import subprocess
from typing import List

from .base import BaseAdapter


class DistributedAdapter(BaseAdapter):
    def __init__(self, local_adapter: BaseAdapter, remote_command: str, remote_user: str, remote_hosts: List[str]):
        self.local_adapter = local_adapter
        self.remote_command = remote_command
        self.remote_user = remote_user
        self.remote_hosts = remote_hosts

    def _remote_command(self, remote_host):
        ssh_process = subprocess.Popen(["ssh", "-T", f"{self.remote_user}@{remote_host}"],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)
        ssh_process.stdin.write(f"screen -dm bash -c \"{self.remote_command}\"")
        ssh_process.stdin.close()

    def _call(self, dataset: np.ndarray):
        # remote call
        for remote_host in self.remote_hosts:
            self._remote_command(remote_host)
        # local call
        return self.local_adapter(dataset)
