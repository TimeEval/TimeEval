import os
import socket
import tempfile
import time
import unittest
from asyncio import Future
from itertools import cycle
from pathlib import Path
from typing import Generator
from typing import List, Union, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import pytest
from durations import Duration
from sklearn.model_selection import ParameterGrid

from tests.fixtures.algorithms import DeviatingFromMean, DeviatingFromMedian
from timeeval import TimeEval, Algorithm, Datasets, RemoteConfiguration, Status
from timeeval.adapters import DockerAdapter
from timeeval.adapters.docker import DockerTimeoutError
from timeeval.remote import Remote
from timeeval.utils.hash_dict import hash_dict

TEST_DOCKER_IMAGE = "mut:5000/akita/timeeval-test-algorithm"


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

    def gather(self, _futures: List[Future], *args, asynchronous=False, **kwargs) -> Union[Generator[Future, None, None], bool]:
        if asynchronous:
            for _ in _futures:
                f = Future()  # type: ignore
                f.set_result(True)
                yield f
        return True

    def close(self) -> None:
        self.closed = True

    def shutdown(self) -> None:
        self.did_shutdown = True


class ExceptionForTest(Exception):
    pass


class MockExceptionClient(MockClient):
    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        f = Future()  # type: ignore
        f.set_exception(ExceptionForTest("test-exception"))
        return f


class MockDockerTimeoutExceptionClient(MockClient):
    def submit(self, task, *args, workers: Optional[List] = None, **kwargs) -> Future:
        f = Future()  # type: ignore
        f.set_exception(DockerTimeoutError("test-exception-timeout"))
        return f


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

    def logs(self):
        return "".encode("utf-8")


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
            Algorithm(name="deviating_from_mean", main=DeviatingFromMean()),
            Algorithm(name="deviating_from_median", main=DeviatingFromMedian(),
                      param_grid=ParameterGrid({"test": [np.int64(2), np.int32(4)]}))
        ]

    @pytest.mark.dask
    def test_distributed_results_and_shutdown_cluster(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                distributed=True, results_path=Path(tmp_path),
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=["localhost"]))
            timeeval.run()

            compare_columns = ["algorithm", "collection", "dataset", "ROC_AUC"]
            results: pd.DataFrame = timeeval.results[compare_columns]
            results = results.groupby(by=["algorithm", "collection", "dataset"])["ROC_AUC"].mean()
            results = pd.DataFrame(results).reset_index()
            pd.testing.assert_frame_equal(
                results,
                self.results.loc[:, compare_columns])

            # wait a bit before testing if all processes have died to allow for slight delay
            time.sleep(0.5)
            processes = [p for p in psutil.process_iter()]
            self.assertFalse(
                any(
                    ("distributed.cli.dask_worker" in c) or ("distributed.cli.dask_scheduler" in c)
                    for p in processes for c in p.cmdline()
                ),
                msg="Not all dask processes were correctly shut down. Still running processes: "
                    f"{[p.cmdline() for p in processes]}"
            )

    @pytest.mark.dask
    def test_run_on_all_hosts(self):
        def _test_func(*args, **kwargs):
            a = time.time_ns()
            os.mkdir(args[0] / str(a))

        with tempfile.TemporaryDirectory() as tmp_path:
            remote = Remote(
                remote_config=RemoteConfiguration(
                    scheduler_host="localhost",
                    worker_hosts=["localhost", "localhost"],
                    remote_python=os.popen("which python").read().rstrip("\n")
                ))
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
                                distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="test-host", worker_hosts=["test-host2"]),
                                results_path=Path(tmp_path),
                                n_jobs=1)
            timeeval.run()

            self.assertTrue(
                (timeeval.results_path / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists()
            )
            self.assertTrue(timeeval.remote.client.closed)
            self.assertTrue(timeeval.remote.client.did_shutdown)
            target_path = timeeval.results_path  # == "/results/YYYY_mm_dd_hh_mm"
            self.assertListEqual(rsync.params[0], ["rsync", "-a", "test-host2:" + str(target_path) + "/", str(target_path)])

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

            self.assertTrue(
                (timeeval.results_path / "docker" / hash_dict({}) / "custom" / "dataset.1" / "1").exists()
            )

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
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [], [], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path),
                                n_jobs=1)
            timeeval._rsync_results()
            self.assertEqual(len(rsync.params), 1)
            self.assertTrue(rsync.params[0], ["rsync", "-a", f"test-host:{tmp_path}/", tmp_path])

    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_catches_future_exception(self, mock_cluster, mock_client, mock_docker, mock_call):
        class Rsync:
            def __init__(self):
                self.params = []

            def __call__(self, *args, **kwargs):
                self.params.append(args[0])

        rsync = Rsync()

        mock_client.return_value = MockExceptionClient()
        mock_cluster.return_value = MockCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = rsync

        datasets = Datasets("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [self.algorithms[0]], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path))
            timeeval.run()
            self.assertListEqual(timeeval.results[["status", "error_message"]].values[0].tolist(), [Status.ERROR, "ExceptionForTest('test-exception')"])

    @patch("timeeval.timeeval.subprocess.call")
    @patch("timeeval.adapters.docker.docker.from_env")
    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_catches_future_timeout_exception(self, mock_cluster, mock_client, mock_docker, mock_call):
        class Rsync:
            def __init__(self):
                self.params = []

            def __call__(self, *args, **kwargs):
                self.params.append(args[0])

        rsync = Rsync()

        mock_client.return_value = MockDockerTimeoutExceptionClient()
        mock_cluster.return_value = MockCluster(workers=2)
        mock_docker.return_value = MockDockerClient()
        mock_call.side_effect = rsync

        datasets = Datasets("./tests/example_data")

        with tempfile.TemporaryDirectory() as tmp_path:
            hosts = [
                socket.gethostname(), "127.0.0.1", socket.gethostbyname(socket.gethostname()), "test-host"
            ]
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [self.algorithms[0]], distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=hosts),
                                results_path=Path(tmp_path))
            timeeval.run()
            self.assertListEqual(timeeval.results[["status", "error_message"]].values[0].tolist(),
                                 [Status.TIMEOUT, "DockerTimeoutError('test-exception-timeout')"])

    @pytest.mark.dask
    @pytest.mark.docker
    def test_catches_future_exception_dask(self):
        datasets = Datasets("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("5 seconds")),
                         data_as_file=True,
                         param_grid=ParameterGrid({"raise": [True], "sleep": [1]}))

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=["localhost"]),
                                results_path=Path(tmp_path))
        timeeval.run()
        status = timeeval.results.loc[0, "status"]
        error_message = timeeval.results.loc[0, "error_message"]

        self.assertEqual(status, Status.ERROR)
        self.assertTrue("Please consider log files" in error_message)

    @pytest.mark.dask
    @pytest.mark.docker
    def test_catches_future_timeout_exception_dask(self):
        datasets = Datasets("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("1 seconds")),
                         data_as_file=True)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=["localhost"]),
                                results_path=Path(tmp_path))
        timeeval.run()

        status = timeeval.results.loc[0, "status"]
        error_message = timeeval.results.loc[0, "error_message"]
        self.assertEqual(status, Status.TIMEOUT)
        self.assertTrue("timed out after" in error_message)

    @pytest.mark.dask
    @pytest.mark.docker
    def test_runs_docker_in_dask(self):
        datasets = Datasets("./tests/example_data")
        algo = Algorithm(name="docker-test-timeout",
                         main=DockerAdapter(TEST_DOCKER_IMAGE, skip_pull=True, timeout=Duration("5 seconds")),
                         data_as_file=True,
                         param_grid=ParameterGrid({"sleep": [1]}))

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, [("test", "dataset-int")], [algo],
                                distributed=True,
                                remote_config=RemoteConfiguration(scheduler_host="localhost", worker_hosts=["localhost"]),
                                results_path=Path(tmp_path))
        timeeval.run()

        status = timeeval.results.loc[0, "status"]
        self.assertEqual(status, Status.OK)
