import os
from pathlib import Path
import tempfile
import unittest
from durations import Duration
from typing import List

from docker.models.containers import Container
import numpy as np
import pytest
import docker

from tests.fixtures.algorithms import DeviatingFromMedian
from tests.fixtures.docker_mocks import TEST_DOCKER_IMAGE
from timeeval.adapters import MultivarAdapter
from timeeval.adapters.docker import DockerAdapter
from timeeval.adapters.multivar import AggregationMethod
from timeeval.data_types import ExecutionType
from timeeval.resource_constraints import ResourceConstraints


class TestMultivarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(100, 3)
        tmp = np.abs(self.X - np.median(self.X, axis=0))
        self.y = tmp / tmp.max(axis=0)
        self.y_median = np.median(self.y, axis=1)
        self.y_mean = np.mean(self.y, axis=1)
        self.y_max = np.max(self.y, axis=1)

        self.X_sum_before = np.sum(self.X, axis=1)
        self.y_sum_before = np.abs(self.X_sum_before - np.median(self.X_sum_before))
        self.y_sum_before /= self.y_sum_before.max()

        self.X = np.c_[self.X, np.zeros(100)]

        self.docker = docker.from_env()
        self.input_data_path = Path("./tests/example_data/dataset.train.csv").absolute()
        self.input_data_multi_path = Path("./tests/example_data/dataset.multi.csv").absolute()

    def tearDown(self) -> None:
        # remove test containers
        containers = self.docker.containers.list(all=True, filters={"ancestor": TEST_DOCKER_IMAGE})
        for c in containers:
            c.remove()
        del self.docker

    def _list_test_containers(self, ancestor_image: str = TEST_DOCKER_IMAGE) -> List[Container]:
        return self.docker.containers.list(all=True, filters={"ancestor": ancestor_image})

    def test_multivar_deviating_from_median_mean(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MEAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_mean, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_median(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MEDIAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_median, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_max(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.MAX)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_max, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_sum_before(self):
        algorithm = MultivarAdapter(DeviatingFromMedian(), AggregationMethod.SUM_BEFORE)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_sum_before, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_raises_on_nested_multivar(self):
        with pytest.raises(AssertionError):
            MultivarAdapter(MultivarAdapter(DeviatingFromMedian()), AggregationMethod.MEAN)

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
            adapter = MultivarAdapter(DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("10 seconds")), AggregationMethod.MEAN)
            res = adapter(self.input_data_path, args)
        np.testing.assert_array_equal(np.zeros(3600), res)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 1)
        self.assertEqual(containers[0].status, "exited")

    @pytest.mark.docker
    def test_multivar_execute_successful(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            args = {
                "executionType": ExecutionType.EXECUTE,
                "resource_constraints": ResourceConstraints.default_constraints(),
                "hyper_params": {"sleep": 0.1},
                "results_path": tmp_path
            }
            adapter = MultivarAdapter(DockerAdapter(TEST_DOCKER_IMAGE, timeout=Duration("10 seconds")), AggregationMethod.MEAN)
            res = adapter(self.input_data_multi_path, args)
        np.testing.assert_array_equal(np.zeros(3600), res)
        containers = self._list_test_containers()
        self.assertEqual(len(containers), 2)
        self.assertEqual(containers[0].status, "exited")
