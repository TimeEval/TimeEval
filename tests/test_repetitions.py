import unittest
import numpy as np
import pandas as pd
from typing import Callable
from pathlib import Path
from itertools import cycle
import tempfile

from timeeval import TimeEval, Algorithm, Datasets


def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
    diffs = np.abs((data - fn(data)))
    diffs = diffs / diffs.max()
    return diffs


def deviating_from_mean(data: np.ndarray, _args) -> np.ndarray:
    return deviating_from(data, np.mean)


def deviating_from_median(data: np.ndarray, _args) -> np.ndarray:
    return deviating_from(data, np.median)


class ErroneousAlgorithm:
    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        if self.count != 2:
            return self.fn(*args, **kwargs)
        else:
            raise ValueError("test error")


class TestRepetitions(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=deviating_from_mean),
            Algorithm(name="deviating_from_median", main=deviating_from_median)
        ]

    def test_multiple_results(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results()

        np.testing.assert_array_almost_equal(results["score_mean"].values, self.results["score"].values)
        self.assertEqual(results["repetitions"].unique()[0], 3)

    def test_return_no_aggregation(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)
        self.assertEqual(len(timeeval.dataset_names) * len(self.algorithms) * 3, len(results))

    def test_error_in_repetition(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)

        algorithms = [
            Algorithm(name="deviating_from_mean", main=ErroneousAlgorithm(deviating_from_mean)),
            Algorithm(name="deviating_from_median", main=deviating_from_median)
        ]
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results()

        np.testing.assert_array_almost_equal(results["score_mean"].values, self.results["score"].values)
        self.assertListEqual(results["repetitions"].unique().tolist(), [2, 3])
