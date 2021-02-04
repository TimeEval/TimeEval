import os
import tempfile
import time
import unittest
from asyncio import Future
from itertools import cycle
from pathlib import Path
from typing import Callable, List, Union, Coroutine, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import socket
import pytest

from timeeval import TimeEval, Algorithm, Datasets
from timeeval.adapters import DockerAdapter, FunctionAdapter
from timeeval.data_types import AlgorithmParameter
from timeeval.remote import Remote
from timeeval.utils.hash_dict import hash_dict


def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
    diffs = np.abs((data - fn(data)))
    diffs = diffs / diffs.max()
    return diffs


def deviating_from_mean(data: AlgorithmParameter, args: dict) -> np.ndarray:
    return deviating_from(data, np.mean)  # type: ignore


def deviating_from_median(data: AlgorithmParameter, args: dict) -> np.ndarray:
    return deviating_from(data, np.median)  # type: ignore


class MockWorker:
    def __init__(self):
        self.address = "localhost"


class MockCluster:
    def __init__(self, workers: int):
        self.scheduler_address = "localhost:8000"
        self.n_workers = workers

    def close(self) -> None:
        pass

    @property
    def workers(self):
        dd = {}
        for i in range(self.n_workers):
            dd[i] = MockWorker()
        return dd


class MockContainer:
    pass


class MockClient:
    def __init__(self):
        self.containers = MockContainer()

    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        result = task(*args, **kwargs)
        f = Future()  # type: ignore
        f.set_result(result)
        return f

    def run(self, task, *args, **kwargs):
        task(*args, **kwargs)

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
    def run(self, image: str, cmd: str, volumes: dict, **kwargs):
        self.image = image
        self.cmd = cmd
        self.volumes = volumes

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(3600, dtype=np.float64).tofile(real_path / Path("anomaly_scores.ts"), sep="\n")

    def prune(self):
        pass


class MockImages:
    def pull(self, image, tag):
        pass


class MockDockerClient:
    def __init__(self):
        self.containers = MockDockerContainer()
        self.images = MockImages()


class TestDistributedTimeEval(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=FunctionAdapter(deviating_from_mean)),
            Algorithm(name="deviating_from_median", main=FunctionAdapter(deviating_from_median))
        ]

    @pytest.mark.dask
    def test_distributed_results_and_shutdown_cluster(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                distributed=True, results_path=Path(tmp_path), ssh_cluster_kwargs={"hosts": ["localhost", "localhost"],
                                                                      "remote_python": os.popen("which python").read().rstrip("\n")})
            timeeval.run()
            np.testing.assert_array_equal(timeeval.results.values[:, :4], self.results.values[:, :4])
            self.assertFalse(any(("distributed.cli.dask_worker" in c) or ("distributed.cli.dask_scheduler" in c)
                                 for p in psutil.process_iter() for c in p.cmdline()))

    @pytest.mark.dask
    def test_run_on_all_hosts(self):
        def _test_func(*args, **kwargs):
            a = time.time_ns()
            os.mkdir(args[0] / str(a))

        with tempfile.TemporaryDirectory() as tmp_path:
            remote = Remote(**{"hosts": ["localhost", "localhost", "localhost"],
                             "remote_python": os.popen("which python").read().rstrip("\n")})
            remote.run_on_all_hosts([(_test_func, [Path(tmp_path)], {})])
            remote.close()
            self.assertEqual(len(os.listdir(tmp_path)), 2)

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
        mock_cluster.return_value = MockCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = rsync

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
        adapter = DockerAdapter("test-image:latest")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())),
                                [Algorithm(name="docker", main=adapter, data_as_file=True)],
                                distributed=True, ssh_cluster_kwargs={"hosts": ["test-host", "test-host2"]},
                                results_path=Path(tmp_path))
            timeeval.run()

            self.assertTrue((Path(tmp_path) / timeeval.start_date / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists())
            self.assertTrue(timeeval.remote.client.closed)
            self.assertTrue(timeeval.remote.client.did_shutdown)
            self.assertListEqual(rsync.params[0], ["rsync", "-a", "test-host:"+str(tmp_path)+"/", str(tmp_path)])
            self.assertListEqual(rsync.params[1], ["rsync", "-a", "test-host2:"+str(tmp_path)+"/", str(tmp_path)])

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_phases(self, mock_docker):
        mock_docker.return_value = MockDockerClient()

        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
        adapter = DockerAdapter("test-image:latest")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())),
                                [Algorithm(name="docker", main=adapter, data_as_file=True)], results_path=Path(tmp_path))
            timeeval.run()

            self.assertTrue((Path(tmp_path) / timeeval.start_date / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists())

    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_aliases_excluded(self, mock_cluster, mock_client, mock_docker, mock_call):
        class Rsync:
            def __init__(self):
                self.params = []

            def __call__(self, *args, **kwargs):
                self.params.append(args[0])

        rsync = Rsync()

        mock_client.return_value = MockClient()
        mock_cluster.return_value = MockCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = rsync

        datasets = Datasets("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [], [], distributed=True,
                                ssh_cluster_kwargs={
                                    "hosts": ["localhost", socket.gethostname(), "127.0.0.1",
                                              socket.gethostbyname(socket.gethostname()), "test-host"]
                                }, results_path=Path(tmp_path))
            timeeval.rsync_results()
            self.assertEqual(len(rsync.params), 1)
            self.assertTrue(rsync.params[0], ["rsync", "-a", f"test-host:{tmp_path}/", tmp_path])
