import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np

from timeeval.adapters import DockerAdapter


class MockDockerContainer:
    def run(self, image: str, cmd: str, volumes: dict, **kwargs):
        self.image = image
        self.cmd = cmd
        self.volumes = volumes

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(10, dtype=np.float64).tofile(real_path / Path("anomaly_scores.ts"), sep="\n")


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
                       '"customParameters": {}, ' \
                       '"executionType": "execute", ' \
                       '"modelInput": null, ' \
                       '"modelOutput": null' \
                       '}\''

        adapter = DockerAdapter("test-image:latest")
        adapter._run_container(Path("/tmp/test.csv"), {"results_path": results_path})

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
