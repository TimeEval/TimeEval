import unittest
import json
import numpy as np
from typing import Callable
from pathlib import Path

from timeeval import TimeEval, Algorithm


def generates_results(dataset) -> dict:
    datasets_config = Path("./tests/example_data/datasets.json")
    datasets = [dataset]

    def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
        diffs = np.abs((data - fn(data)))
        diffs = diffs / diffs.max()

        return diffs

    def deviating_from_mean(data: np.ndarray) -> np.ndarray:
        return deviating_from(data, np.mean)

    def deviating_from_median(data: np.ndarray) -> np.ndarray:
        return deviating_from(data, np.median)

    algorithms = [
        Algorithm(name="deviating_from_mean", function=deviating_from_mean, data_as_file=False),
        Algorithm(name="deviating_from_median", function=deviating_from_median, data_as_file=False)
    ]

    timeeval = TimeEval(datasets, algorithms, dataset_config=datasets_config)
    timeeval.run()
    return timeeval.results


def generates_results_multi(dataset) -> dict:
    datasets_config = Path("./tests/example_data/datasets.json")
    datasets = [dataset]

    def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
        diffs = np.abs((data - fn(data, axis=0)))
        diffs = diffs / diffs.max(axis=0)

        return diffs.mean(axis=1)

    def deviating_from_mean(data: np.ndarray) -> np.ndarray:
        return deviating_from(data, np.mean)

    def deviating_from_median(data: np.ndarray) -> np.ndarray:
        return deviating_from(data, np.median)

    algorithms = [
        Algorithm(name="deviating_from_mean", function=deviating_from_mean, data_as_file=False),
        Algorithm(name="deviating_from_median", function=deviating_from_median, data_as_file=False)
    ]

    timeeval = TimeEval(datasets, algorithms, dataset_config=datasets_config)
    timeeval.run()
    return timeeval.results


class TestImportData(unittest.TestCase):
    def test_generates_correct_results_from_2files(self):
        DATASET = "dataset.1"
        generated_results = generates_results(DATASET)
        true_results = json.load(open("./tests/example_data/results.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET]["auroc"])

    def test_generates_correct_results_from_1file(self):
        DATASET = "dataset.3"
        generated_results = generates_results(DATASET)
        true_results = json.load(open("./tests/example_data/results.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET]["auroc"])

    def test_generates_correct_results_from_multi_file(self):
        DATASET = "dataset.4"
        generated_results = generates_results_multi(DATASET)
        true_results = json.load(open("./tests/example_data/results_multi.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET]["auroc"])


if __name__ == "__main__":
    unittest.main()
