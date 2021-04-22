import getpass
import logging
import subprocess
from typing import List

from .base import Adapter
from ..data_types import TSFunction, AlgorithmParameter


class DistributedAdapter(Adapter):
    """
    Please, be aware that you need password-less ssh to the remote machines!
    """

    def __init__(self, algorithm: TSFunction, remote_command: str, remote_user: str, remote_hosts: List[str]):
        self.algorithm = algorithm
        self.remote_command = remote_command
        self.remote_user = remote_user
        self.remote_hosts = remote_hosts

    def _remote_command(self, remote_host):
        current_user = getpass.getuser()
        if self.remote_user != current_user:
            logging.warning(f"You are currently running this Python Script as user '{current_user}', "
                            f"but you are trying to connect to '{remote_host}' as user '{self.remote_user}'.")

        ssh_process = subprocess.Popen(["ssh", "-T", f"{self.remote_user}@{remote_host}"],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)
        ssh_process.stdin.write(f"screen -dm bash -c \"{self.remote_command}\"")
        ssh_process.stdin.close()

    def _call(self, dataset: AlgorithmParameter, args: dict):
        # remote call
        for remote_host in self.remote_hosts:
            self._remote_command(remote_host)
        # local call
        return self.algorithm(dataset, args)
