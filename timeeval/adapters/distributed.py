import getpass
import logging
import subprocess
from typing import List, Any, Dict

from .base import Adapter
from ..data_types import TSFunction, AlgorithmParameter


class DistributedAdapter(Adapter):
    """
    An adapter that allows to run a function as an anomaly detector on multiple remote machines. So far, this adapter
    only supports TSFunctions as algorithms.
    Please, be aware that you need password-less ssh to the remote machines!

    .. warning::
        This adapter is deprecated and will be removed in a future version of TimeEval.

    Parameters
    ----------

    algorithm : TSFunction
        The function to run.

    remote_command : str
        The command to run on the remote machines.

    remote_user : str
        The user to use for the ssh connection.

    remote_hosts : List[str]
        The hosts to connect to.
    """

    def __init__(self, algorithm: TSFunction, remote_command: str, remote_user: str, remote_hosts: List[str]) -> None:
        self.algorithm = algorithm
        self.remote_command = remote_command
        self.remote_user = remote_user
        self.remote_hosts = remote_hosts

    def _remote_command(self, remote_host: str) -> None:
        current_user = getpass.getuser()
        if self.remote_user != current_user:
            logging.warning(f"You are currently running this Python Script as user '{current_user}', "
                            f"but you are trying to connect to '{remote_host}' as user '{self.remote_user}'.")

        ssh_process = subprocess.Popen(["ssh", "-T", f"{self.remote_user}@{remote_host}"],
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True,
                                       bufsize=0)
        stdin = ssh_process.stdin
        if stdin is not None:
            stdin.write(f"screen -dm bash -c \"{self.remote_command}\"")
            stdin.close()

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        # remote call
        for remote_host in self.remote_hosts:
            self._remote_command(remote_host)
        # local call
        return self.algorithm(dataset, args)
