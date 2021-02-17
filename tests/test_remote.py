import pytest
import unittest

from dask.distributed import SSHCluster
from timeeval.remote import Remote, RemoteConfiguration


class TestRemote(unittest.TestCase):
    @pytest.mark.dask
    def test_restart_already_running(self):
        already_running_cluster = SSHCluster(hosts=["localhost"])
        with self.assertLogs("Remote", level="WARNING") as logger:
            new_remote = Remote(remote_config=RemoteConfiguration(scheduler_host="localhost"))
            self.assertTrue("restart cluster" in logger.output[0])
            already_running_cluster.close()
            new_remote.close()

    @pytest.mark.dask
    def test_other_than_already_running_exception(self):
        with self.assertRaises(ValueError):
            Remote(remote_config=RemoteConfiguration(scheduler_host=None))
