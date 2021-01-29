import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List, Union, Coroutine, Optional
from asyncio import Future
from itertools import cycle
import tempfile

from timeeval import TimeEval, Algorithm, Datasets
from timeeval.adapters import DockerAdapter


def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
    diffs = np.abs((data - fn(data)))
    diffs = diffs / diffs.max()
    return diffs


def deviating_from_mean(data: np.ndarray, args) -> np.ndarray:
    return deviating_from(data, np.mean)


def deviating_from_median(data: np.ndarray, args) -> np.ndarray:
    return deviating_from(data, np.median)


class MockCluster:
    def __init__(self):
        self.scheduler_address = "localhost:8000"

    def close(self) -> None:
        pass


class MockContainer:
    def prune(self):
        pass


class MockClient:
    def __init__(self):
        self.containers = MockContainer()

    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        result = task(*args, **kwargs)
        f = Future()
        f.set_result(result)
        return f

    def gather(self, _futures: List[Future], *args, asynchronous=False, **kwargs) -> Union[Coroutine, bool]:
        if asynchronous:
            return self._gather(_futures, *args, **kwargs)
        return True

    async def _gather(self, _futures: List[Future], *args, **kwargs):
        return True

    def close(self) -> None:
        self.closed = True

    def shutdown(self) -> None:
        self.did_shutdown = True


class MockDockerContainer:
    def run(self, image: str, cmd: str, volumes: dict):
        self.image = image
        self.cmd = cmd
        self.volumes = volumes

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(3600, dtype=np.float).tofile(real_path / Path("anomaly_scores.ts"), sep="\n")


class MockImages:
    def pull(self, image, tag):
        self.pulled = True


class MockDockerClient:
    def __init__(self):
        self.containers = MockDockerContainer()
        self.images = MockImages()


class TestDistributedTimeEval(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=deviating_from_mean),
            Algorithm(name="deviating_from_median", main=deviating_from_median)
        ]

    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_distributed_results(self, mock_cluster, mock_client, mock_docker):
        mock_client.return_value = MockClient()
        mock_cluster.return_value = MockCluster()
        mock_docker.return_value = MockDockerClient()

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                distributed=True, ssh_cluster_kwargs={"hosts": ["test-host"]}, results_path=Path(tmp_path))
            timeeval.run()
        np.testing.assert_array_equal(timeeval.results.values[:, :4], self.results.values[:, :4])

    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_distributed_phases(self, mock_cluster, mock_client, mock_docker, mock_call):
        class Rsync:
            def __init__(self):
                self.params = []

            def __call__(self, *args, **kwargs):
                self.params.append(args[0])

        rsync = Rsync()

        mock_client.return_value = MockClient()
        mock_cluster.return_value = MockCluster()
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = rsync

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
        adapter = DockerAdapter("test-image:latest")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())),
                                [Algorithm(name="docker", main=adapter, data_as_file=True)],
                                distributed=True, ssh_cluster_kwargs={"hosts": ["test-host", "test-host2"]}, results_path=Path(tmp_path))
            timeeval.run()

            self.assertTrue(adapter.client.images.pulled)
            self.assertTrue((Path(tmp_path) / "docker" / "custom" / "dataset.1").exists())
            self.assertTrue(timeeval.remote.client.closed)
            self.assertTrue(timeeval.remote.client.did_shutdown)
            self.assertListEqual(rsync.params[0], ["rsync", "-a", str(tmp_path)+"/", "test-host:"+str(tmp_path)])
            self.assertListEqual(rsync.params[1], ["rsync", "-a", str(tmp_path) + "/", "test-host2:" + str(tmp_path)])
