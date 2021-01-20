import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List
from asyncio import Future

from timeeval import TimeEval, Algorithm


def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
    diffs = np.abs((data - fn(data)))
    diffs = diffs / diffs.max()
    return diffs


def deviating_from_mean(data: np.ndarray) -> np.ndarray:
    return deviating_from(data, np.mean)


def deviating_from_median(data: np.ndarray) -> np.ndarray:
    return deviating_from(data, np.median)


class MockCluster:
    def close(self) -> None:
        pass


class MockClient:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, task, *args) -> Future:
        result = task(*args)
        f = Future()
        f.set_result(result)
        return f

    def gather(self, _futures: List[Future]) -> None:
        pass

    def close(self) -> None:
        pass


class TestTimeEval(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithm = [
            Algorithm(name="deviating_from_mean", function=deviating_from_mean, data_as_file=False),
            Algorithm(name="deviating_from_median", function=deviating_from_median, data_as_file=False)
        ]

    @patch("timeeval.remote.Client")
    @patch("timeeval.remote.SSHCluster")
    def test_distributed_results(self, mock_cluster, mock_client):
        mock_client.return_value = MockClient()
        mock_cluster.return_value = MockCluster()

        timeeval = TimeEval(self.results.dataset.unique(), self.algorithm,
                            dataset_config=Path("tests/example_data/datasets.json"),
                            distributed=True)
        timeeval.run()
        np.testing.assert_array_equal(timeeval.results.values[:, :-3], self.results.values[:, :-3])

