import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

import docker
import numpy as np
import psutil
import pytest
from docker.models.containers import Container
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
        input_string = ('execute-algorithm \'{'
                        f'"dataInput": "{(DATASET_TARGET_PATH / "test.csv").resolve()}", '
                        f'"dataOutput": "{(RESULTS_TARGET_PATH / SCORES_FILE_NAME).resolve()}", '
                        f'"modelInput": "{(RESULTS_TARGET_PATH / MODEL_FILE_NAME).resolve()}", '
                        f'"modelOutput": "{(RESULTS_TARGET_PATH / MODEL_FILE_NAME).resolve()}", '
                        '"executionType": "train", "customParameters": {"a": 0}'
                        '}\'')

        adapter = DockerAdapter("test-image")
        adapter._run_container(Path("/tmp/test.csv"), {
            "results_path": results_path,
            "hyper_params": {"a": 0},
            "executionType": ExecutionType.TRAIN
        })

        self.assertEqual(mock_docker_client.containers.cmd, input_string)

    @patch("timeeval.adapters.docker.docker.from_env")
    def test_results_at_correct_location(self, mock_client):
        docker_mock = MockDockerClient(write_scores_file=True)
        mock_client.return_value = docker_mock

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
        docker_mock = MockDockerClient(write_scores_file=True)
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
        docker_mock = MockDockerClient(write_scores_file=True)
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


class TestDockerAdapterDocker(unittest.TestCase):
    def setUp(self) -> None:
        self.docker = docker.from_env()
        self.input_data_path = Path("./tests/example_data/dataset.train.csv").absolute()

    def tearDown(self) -> None:
        # remove test containers
        containers = self.docker.containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        for c in containers:
            c.remove()
        del self.docker

    def _list_test_containers(self, ancestor_image: str = TEST_DOCKER_IMAGE) -> List[Container]:
        return self.docker.containers.list(all=True, filters={"ancestor": ancestor_image})

    @pytest.mark.docker
    def test_execute_successful(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.EXECUTE,
                "resource_constraints": ResourceConstraints.default_constraints(),
                "hyper_params": {"sleep": 0.1},
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("10 seconds"))
            res = adapter(self.input_data_path, args)
        np.testing.assert_array_equal(np.zeros(3600), res)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_execute_timeout(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "results_path": tmp_path
            }
            # unit typo in Durations lib (https://github.com/oleiade/durations/blob/master/durations/constants.py#L34)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            with self.assertRaises(DockerTimeoutError):
                adapter(self.input_data_path, args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_execute_timeout_2(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "results_path": tmp_path,
                "resource_constraints": ResourceConstraints(execute_timeout=Duration("100 miliseconds"))
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE)
            with self.assertRaises(DockerTimeoutError):
                adapter(self.input_data_path, args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_execute_timeout_if_no_scores(self):
        args = {
            "executionType": ExecutionType.EXECUTE,
            "resource_constraints": ResourceConstraints(use_preliminary_scores_on_execute_timeout=False)
        }
        adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
        with self.assertRaises(DockerTimeoutError):
            adapter(Path("dummy"), args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_execute_ignore_timeout_if_scores(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.EXECUTE,
                "resource_constraints": ResourceConstraints(use_preliminary_scores_on_execute_timeout=True),
                "hyper_params": {"write_prelim_results": True, "sleep": 20},
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("5 seconds"))
            res = adapter(self.input_data_path, args)
        np.testing.assert_array_equal(np.zeros(3600), res)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_execute_timeout_besides_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            # add a model file from potentially previous training
            (tmp_path / MODEL_FILE_NAME).touch()
            args = {
                "executionType": ExecutionType.EXECUTE,
                "resource_constraints": ResourceConstraints(use_preliminary_scores_on_execute_timeout=False),
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            with self.assertRaises(DockerTimeoutError):
                adapter(self.input_data_path, args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_successful(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.TRAIN,
                "resource_constraints": ResourceConstraints.default_constraints(),
                "hyper_params": {"sleep": 0.1},
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("10 seconds"))
            result = adapter(self.input_data_path, args)

        # TEST_DOCKER_IMAGE does not write a model file, therefore, we don't assert its existence!
        self.assertEqual(result, self.input_data_path)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_timeout(self):
        args = {
            "executionType": ExecutionType.TRAIN,
            "resource_constraints": ResourceConstraints(use_preliminary_model_on_train_timeout=False)
        }
        adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
        with self.assertRaises(DockerTimeoutError):
            adapter(Path("dummy"), args)

        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_timeout_2(self):
        args = {
            "executionType": ExecutionType.TRAIN,
            "resource_constraints": ResourceConstraints(
                use_preliminary_model_on_train_timeout=False,
                train_timeout=Duration("100 miliseconds")
            )
        }
        adapter = DockerAdapter(TEST_DOCKER_IMAGE)
        with self.assertRaises(DockerTimeoutError):
            adapter(Path("dummy"), args)

        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_timeout_if_no_model(self):
        # TEST_DOCKER_IMAGE does not write a model file!
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.TRAIN.value,
                "resource_constraints": ResourceConstraints(use_preliminary_model_on_train_timeout=True),
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            with self.assertRaises(DockerTimeoutError) as e:
                adapter(Path("dummy"), args)

        containers = self._list_test_containers()
        self.assertIn("not build a model", str(e.exception))
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_train_ignore_timeout_if_model(self):
        dummy_path = Path("dummy")
        # TEST_DOCKER_IMAGE does not write a model file!
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            # create model file to trick adapter into thinking that training was successful
            (tmp_path / MODEL_FILE_NAME).touch()
            args = {
                "executionType": ExecutionType.TRAIN,
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("100 miliseconds"))
            res = adapter(dummy_path, args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")
        self.assertEqual(res, dummy_path)

    @pytest.mark.docker
    def test_train_ignore_timeout_if_model_2(self):
        dummy_path = Path("dummy")
        # TEST_DOCKER_IMAGE does not write a model file!
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            # create model file to trick adapter into thinking that training was successful
            (tmp_path / MODEL_FILE_NAME).touch()
            args = {
                "executionType": ExecutionType.TRAIN.value,
                "resource_constraints": ResourceConstraints(
                    use_preliminary_model_on_train_timeout=True,
                    train_timeout=Duration("100 miliseconds")
                ),
                "results_path": tmp_path
            }
            adapter = DockerAdapter(TEST_DOCKER_IMAGE)
            res = adapter(dummy_path, args)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")
        self.assertEqual(res, dummy_path)

    @pytest.mark.docker
    def test_algorithm_error(self):
        with self.assertRaises(DockerAlgorithmFailedError):
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("1 minute"))
            adapter(Path("dummy"), {"hyper_params": {"raise": True}})
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_faster_than_timeout(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("30 seconds"))
            result = adapter(self.input_data_path, {"results_path": tmp_path, "hyper_params": {"sleep": 1}})
        np.testing.assert_array_equal(np.zeros(3600), result)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_prune(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            adapter = DockerAdapter(TEST_DOCKER_IMAGE)
            _ = adapter(self.input_data_path, {"results_path": tmp_path, "hyper_params": {"sleep": 0.1}})
        self.assertEqual(len(self._list_test_containers()), 1)
        adapter.get_finalize_fn()()
        self.assertListEqual(self._list_test_containers(), [])
