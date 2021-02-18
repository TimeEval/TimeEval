import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import docker
import numpy as np
import psutil
import pytest
from durations import Duration

from timeeval.adapters import DockerAdapter
from timeeval.adapters.docker import DATASET_TARGET_PATH, RESULTS_TARGET_PATH, SCORES_FILE_NAME, MODEL_FILE_NAME
from timeeval.adapters.docker import DockerTimeoutError, DockerAlgorithmFailedError

DUMMY_CONTAINER = "algorithm-template-dummy"
TEST_IMAGE = "mut:5000/akita/timeeval-test-algorithm"


class MockDockerContainer:
    def wait(self, timeout=None):
        return {"Error": None, "StatusCode": 0}

    def run(self, image: str, cmd: str, volumes: dict, **kwargs):
        self.image = image
        self.cmd = cmd
        self.volumes = volumes
        self.run_kwargs = kwargs

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(10, dtype=np.float64).tofile(real_path / Path(SCORES_FILE_NAME), sep="\n")
        return self

    def logs(self):
        return "".encode("utf-8")


class MockDockerClient:
    def __init__(self):
        self.containers = MockDockerContainer()


class TestDockerAdapter(unittest.TestCase):
    @patch("timeeval.adapters.docker.docker.from_env")
    def test_correct_json_string(self, mock_client):
        mock_docker_client = MockDockerClient()
        mock_client.return_value = mock_docker_client
        results_path = Path("./results/")
        input_string = 'execute-algorithm \'{' \
                       f'"dataInput": "{DATASET_TARGET_PATH}/test.csv", ' \
                       f'"dataOutput": "{RESULTS_TARGET_PATH}/{SCORES_FILE_NAME}", ' \
                       f'"modelInput": "{RESULTS_TARGET_PATH}/{MODEL_FILE_NAME}", ' \
                       f'"modelOutput": "{RESULTS_TARGET_PATH}/{MODEL_FILE_NAME}", ' \
                       '"customParameters": {"a": 0}, ' \
                       '"executionType": "execute"' \
                       '}\''

        adapter = DockerAdapter("test-image:latest")
        adapter._run_container(Path("/tmp/test.csv"), {"results_path": results_path, "hyper_params": {"a": 0}})

        self.assertEqual(mock_docker_client.containers.cmd, input_string)

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_results_at_correct_location(self, mock_client):
        mock_docker_client = MockDockerClient()
        mock_client.return_value = mock_docker_client

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image:latest")
            result = adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})
        np.testing.assert_array_equal(result, np.arange(10, dtype=np.float64))

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_assertion_error(self, mock_client):
        mock_docker_client = MockDockerClient()
        mock_client.return_value = mock_docker_client

        with self.assertRaises(AssertionError):
            adapter = DockerAdapter("test-image:latest")
            adapter(np.random.rand(10), {})

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_sets_default_resource_constraints(self, mock_client):
        docker_mock = MockDockerClient()
        mock_client.return_value = docker_mock

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image:latest")
            adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})

        run_kwargs = docker_mock.containers.run_kwargs
        self.assertIn("mem_swappiness", run_kwargs, msg="mem_swappiness was not set by DockerAdapter")
        self.assertIn("mem_limit", run_kwargs, msg="mem_limit was not set by DockerAdapter")
        self.assertIn("memswap_limit", run_kwargs, msg="memswap_limit was not set by DockerAdapter")
        self.assertIn("nano_cpus", run_kwargs, msg="nano_cpus was not set by DockerAdapter")

        # must always disable swapping:
        self.assertEqual(run_kwargs["mem_swappiness"], 0)
        # no swap means mem_limit and memswap_limit must be the same!
        self.assertEqual(run_kwargs["mem_limit"], run_kwargs["memswap_limit"])
        # must use all available CPUs
        self.assertEqual(run_kwargs["nano_cpus"], multiprocessing.cpu_count()*1e9)

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_overwrite_resource_constraints(self, mock_client):
        docker_mock = MockDockerClient()
        mock_client.return_value = docker_mock
        mem_overwrite = 500
        cpu_overwrite = 0.25

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image:latest",
                                    memory_limit_overwrite=mem_overwrite,
                                    cpu_limit_overwrite=cpu_overwrite
                                    )
            adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})

        run_kwargs = docker_mock.containers.run_kwargs
        self.assertEqual(run_kwargs["mem_limit"], mem_overwrite)
        self.assertEqual(run_kwargs["memswap_limit"], mem_overwrite)
        self.assertEqual(run_kwargs["nano_cpus"], cpu_overwrite*1e9)

    @pytest.mark.docker
    def test_timeout_docker(self):
        with self.assertRaises(DockerTimeoutError):
            adapter = DockerAdapter(TEST_IMAGE, timeout=Duration("100 miliseconds"))
            adapter(Path("dummy"))
        self.assertListEqual(docker.from_env().containers.list(all=True, filters={"name": TEST_IMAGE}), [])

    @pytest.mark.docker
    def test_algorithm_error_docker(self):
        with self.assertRaises(DockerAlgorithmFailedError):
            adapter = DockerAdapter(TEST_IMAGE, timeout=Duration("1 minute"))
            adapter(Path("dummy"), {"hyper_params": {"raise": True}})
        self.assertListEqual(docker.from_env().containers.list(all=True, filters={"name": TEST_IMAGE}), [])

    @pytest.mark.docker
    def test_faster_than_timeout_docker(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter(TEST_IMAGE, timeout=Duration("1 minute, 40 seconds"))
            result = adapter(Path("./tests/example_data/dataset.train.csv").absolute(), {"result_path": tmp_path, "hyper_params": {}})
            np.testing.assert_array_equal(np.zeros(3600), result)
        self.assertListEqual(docker.from_env().containers.list(all=True, filters={"name": TEST_IMAGE}), [])
