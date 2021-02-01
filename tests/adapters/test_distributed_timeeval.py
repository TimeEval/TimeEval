import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List
from asyncio import Future
from itertools import cycle
import tempfile
import os

from timeeval import TimeEval, Algorithm, Datasets


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


class MockClient:
    def __init__(self, *args, **kwargs):
        pass

    def submit(self, task, *args) -> Future:
        result = task(*args)
        f = Future()
        f.set_result(result)
        return f

    async def gather(self, _futures: List[Future], *args, **kwargs):
        return True

    def close(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class TestDistributedTimeEval(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=deviating_from_mean),
            Algorithm(name="deviating_from_median", main=deviating_from_median)
        ]

    @unittest.skipIf(os.getenv("CI"), reason="CI test runs in a slim Docker container and does not support SSH-connections")
    def test_distributed_results_and_shutdown_cluster(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                distributed=True, results_path=Path(tmp_path), ssh_cluster_kwargs={"hosts": ["localhost", "localhost"],
                                                                      "remote_python": os.popen("which python").read().rstrip("\n")})
            timeeval.run()
        np.testing.assert_array_equal(timeeval.results.values[:, :4], self.results.values[:, :4])

        self.assertEqual(os.popen("pgrep -f distributed.cli.dask").read(), "")



