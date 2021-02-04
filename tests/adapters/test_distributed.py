import unittest
from unittest.mock import patch

import numpy as np

from tests.fixtures.algorithms import DeviatingFromMedian
from timeeval.adapters import DistributedAdapter


class SSHProcess:
    def __init__(self):
        class STDin:
            def __init__(self):
                self.written = list()

            def write(self, cmd):
                self.written.append(cmd)

            def close(self):
                pass

        self.stdin = STDin()


class TestDistributedAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.random.rand(1000, 10)
        self.remote_command = "test_command"
        self.remote_user = "test_user"
        self.remote_hosts = [
            "testhost01",
            "testhost02"
        ]

    @patch('subprocess.Popen')
    def test_screen_command(self, mock_A):
        cmd = f'screen -dm bash -c "{self.remote_command}"'

        ssh_processes = [SSHProcess(), SSHProcess()]
        mock_A.side_effect = ssh_processes

        algorithm = DistributedAdapter(DeviatingFromMedian(),
                                       remote_command=self.remote_command,
                                       remote_user=self.remote_user,
                                       remote_hosts=self.remote_hosts)

        algorithm(self.X)

        for p in range(len(ssh_processes)):
            self.assertEqual(len(ssh_processes[p].stdin.written), 1)
            self.assertEqual(ssh_processes[p].stdin.written[0], cmd)
