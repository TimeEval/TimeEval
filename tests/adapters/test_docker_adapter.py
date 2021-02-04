import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import docker
import numpy as np
import pytest
from durations import Duration

from timeeval.adapters import DockerAdapter
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

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(10, dtype=np.float64).tofile(real_path / Path("anomaly_scores.ts"), sep="\n")
        return self


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
                       '"dataInput": "/data/test.csv", ' \
                       '"dataOutput": "/results/anomaly_scores.ts", ' \
                       '"modelInput": "/results/model.pkl", ' \
                       '"modelOutput": "/results/model.pkl", ' \
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
            result = adapter(Path(tmp_path))
            np.testing.assert_array_equal(np.zeros(3600), result)
        self.assertListEqual(docker.from_env().containers.list(all=True, filters={"name": TEST_IMAGE}), [])
