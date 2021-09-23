import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import docker
import numpy as np
import psutil
import pytest
from durations import Duration

from tests.fixtures.docker_mocks import MockDockerClient, TEST_DOCKER_IMAGE
from timeeval import ResourceConstraints
from timeeval.adapters import DockerAdapter
from timeeval.adapters.docker import (
    DATASET_TARGET_PATH,
    RESULTS_TARGET_PATH,
    SCORES_FILE_NAME,
    MODEL_FILE_NAME,
    DockerTimeoutError,
    DockerAlgorithmFailedError,
    AlgorithmInterface
)
from timeeval.data_types import ExecutionType


class TestDockerAdapter(unittest.TestCase):

    def test_algorithm_interface_with_numpy(self):
        ai = AlgorithmInterface(
            dataInput=Path("in.csv"),
            dataOutput=Path("out.csv"),
            modelInput=Path("model-in.csv"),
            modelOutput=Path("model-out.csv"),
            executionType=ExecutionType.TRAIN,
            customParameters={
                "bool": True,
                "float": 1e-3,
                "numpy": [np.int64(4), np.float64(2.3)],
                "numpy-list": np.arange(2)
            }
        )
        ai_string = ('{"dataInput": "in.csv", "dataOutput": "out.csv", "modelInput": "model-in.csv", '
                     '"modelOutput": "model-out.csv", "executionType": "train", "customParameters": '
                     '{"bool": true, "float": 0.001, "numpy": [4, 2.3], "numpy-list": [0, 1]}}')
        self.assertEqual(ai.to_json_string(), ai_string)

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
                       '"executionType": "train", ' \
                       '"customParameters": {"a": 0}' \
                       '}\''

        adapter = DockerAdapter("test-image")
        adapter._run_container(Path("/tmp/test.csv"), {
            "results_path": results_path,
            "hyper_params": {"a": 0},
            "executionType": ExecutionType.TRAIN
        })

        self.assertEqual(mock_docker_client.containers.cmd, input_string)

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_results_at_correct_location(self, mock_client):
        mock_docker_client = MockDockerClient()
        mock_client.return_value = mock_docker_client

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image")
            result = adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})
        np.testing.assert_array_equal(result, np.arange(10, dtype=np.float64))

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_assertion_error(self, mock_client):
        mock_docker_client = MockDockerClient()
        mock_client.return_value = mock_docker_client

        with self.assertRaises(AssertionError):
            adapter = DockerAdapter("test-image")
            adapter(np.random.rand(10), {})

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_sets_default_resource_constraints(self, mock_client):
        docker_mock = MockDockerClient()
        mock_client.return_value = docker_mock

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image")
            adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})

        self.assertTrue(docker_mock.containers.stopped)

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
        self.assertEqual(run_kwargs["nano_cpus"], psutil.cpu_count() * 1e9)

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_overwrite_resource_constraints(self, mock_client):
        docker_mock = MockDockerClient()
        mock_client.return_value = docker_mock
        mem_overwrite = 500
        cpu_overwrite = 0.25

        with tempfile.TemporaryDirectory() as tmp_path:
            adapter = DockerAdapter("test-image",
                                    memory_limit_overwrite=mem_overwrite,
                                    cpu_limit_overwrite=cpu_overwrite
                                    )
            adapter(Path("tests/example_data/data.txt"), {"results_path": Path(tmp_path)})

        run_kwargs = docker_mock.containers.run_kwargs
        self.assertEqual(run_kwargs["mem_limit"], mem_overwrite)
        self.assertEqual(run_kwargs["memswap_limit"], mem_overwrite)
        self.assertEqual(run_kwargs["nano_cpus"], cpu_overwrite * 1e9)

    @pytest.mark.docker
    def test_timeout_docker(self):
        with self.assertRaises(DockerTimeoutError):
            # unit typo in Durations lib (https://github.com/oleiade/durations/blob/master/durations/constants.py#L34)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            adapter(Path("dummy"))
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_timeout_docker(self):
        args = {
            "executionType": ExecutionType.TRAIN,
            "resource_constraints": ResourceConstraints(train_fails_on_timeout=True)
        }
        with self.assertRaises(DockerTimeoutError):
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            adapter(Path("dummy"), args)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_raise_train_timeout_if_no_model_docker(self):
        # TEST_DOCKER_IMAGE does not write a model file!
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.TRAIN.value,
                # "resource_constraints": ResourceConstraints.no_constraints(),
                "results_path": tmp_path
            }
            dummy_path = Path("dummy")
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))

            with self.assertRaises(DockerTimeoutError) as e:
                adapter(dummy_path, args)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertIn("not build a model", str(e.exception))
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_ignore_train_timeout_docker(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            # create model file to trick adapter into thinking that training was successful
            (tmp_path / MODEL_FILE_NAME).touch()
            args = {
                "executionType": ExecutionType.TRAIN,
                # "resource_constraints": ResourceConstraints.no_constraints(),
                "results_path": tmp_path
            }
            dummy_path = Path("dummy")
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            res = adapter(dummy_path, args)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")
        self.assertEqual(res, dummy_path)

    @pytest.mark.docker
    def test_ignore_train_timeout_docker_2(self):
        args = {
            "executionType": ExecutionType.TRAIN.value,
            "resource_constraints": ResourceConstraints(
                train_fails_on_timeout=False,
                train_timeout=Duration("100 miliseconds")
            )
        }
        dummy_path = Path("dummy")
        adapter = DockerAdapter(TEST_DOCKER_IMAGE)
        res = adapter(dummy_path, args)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")
        self.assertEqual(res, dummy_path)

    @pytest.mark.docker
    def test_ignore_train_timeout_docker_2(self):
        args = {
            "executionType": ExecutionType.TRAIN.value,
            "resource_constraints": ResourceConstraints(
                train_fails_on_timeout=False,
                train_timeout=Duration("100 miliseconds")
            )
        }
        dummy_path = Path("dummy")
        adapter = DockerAdapter(TEST_DOCKER_IMAGE)
        res = adapter(dummy_path, args)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")
        self.assertEqual(res, dummy_path)

    @pytest.mark.docker
    def test_algorithm_error_docker(self):
        with self.assertRaises(DockerAlgorithmFailedError):
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("1 minute"))
            adapter(Path("dummy"), {"hyper_params": {"raise": True}})
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_faster_than_timeout_docker(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("1 minute, 40 seconds"))
            result = adapter(Path("./tests/example_data/dataset.train.csv").absolute(),
                             {"results_path": tmp_path, "hyper_params": {}})
            np.testing.assert_array_equal(np.zeros(3600), result)
        containers = docker.from_env().containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_prune_docker(self):
        docker_client = docker.from_env()
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE)
            _ = adapter(Path("./tests/example_data/dataset.train.csv").absolute(),
                        {"results_path": tmp_path, "hyper_params": {"sleep": 2}})
        containers = docker_client.containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        # remove containers before assertions to make sure that they are gone in the case of failing assertions
        for c in containers:
            c.remove()
        self.assertEqual(len(containers), 1)
        adapter.get_finalize_fn()()
        self.assertListEqual(docker_client.containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE}), [])
