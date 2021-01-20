import unittest
import json
import numpy as np
from typing import Callable
from pathlib import Path

from timeeval import TimeEval, Algorithm, Datasets


def generates_results(dataset) -> dict:
    datasets_config = Path("./tests/example_data/datasets.json")

    def deviating_from(fn: Callable) -> Callable[[np.ndarray], np.ndarray]:
        def call(data: np.ndarray) -> np.ndarray:
            diffs = np.abs((data - fn(data)))
            diffs = diffs / diffs.max()

            return diffs
        return call

    algorithms = [
        Algorithm(name="deviating_from_mean", function=deviating_from(np.mean), data_as_file=False),
        Algorithm(name="deviating_from_median", function=deviating_from(np.median), data_as_file=False)
    ]

    datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
    timeeval = TimeEval(datasets, [dataset], algorithms)
    timeeval.run()
    return timeeval.results


def generates_results_multi(dataset) -> dict:
    datasets_config = Path("./tests/example_data/datasets.json")

    def deviating_from(fn: Callable) -> Callable[[np.ndarray], np.ndarray]:
        def call(data: np.ndarray) -> np.ndarray:
            diffs = np.abs((data - fn(data, axis=0)))
            diffs = diffs / diffs.max(axis=0)

            return diffs.mean(axis=1)
        return call

    algorithms = [
        Algorithm(name="deviating_from_mean", function=deviating_from(np.mean), data_as_file=False),
        Algorithm(name="deviating_from_median", function=deviating_from(np.median), data_as_file=False)
    ]

    datasets = Datasets("./tests/example_data", custom_datasets_file=datasets_config)
    timeeval = TimeEval(datasets, [dataset], algorithms)
    timeeval.run()
    return timeeval.results


class TestImportData(unittest.TestCase):
    def test_generates_correct_results_from_2files(self):
        DATASET = ("custom", "dataset.1")
        generated_results = generates_results(DATASET)
        true_results = json.load(open("./tests/example_data/results.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET[1]]["auroc"])

    def test_generates_correct_results_from_1file(self):
        DATASET = ("custom", "dataset.3")
        generated_results = generates_results(DATASET)
        true_results = json.load(open("./tests/example_data/results.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET[1]]["auroc"])

    def test_generates_correct_results_from_multi_file(self):
        DATASET = ("custom", "dataset.4")
        generated_results = generates_results_multi(DATASET)
        true_results = json.load(open("./tests/example_data/results_multi.json", "r"))

        for algorithm in ["deviating_from_mean", "deviating_from_median"]:
            self.assertEqual(generated_results[algorithm][DATASET]["auroc"],
                             true_results[algorithm][DATASET[1]]["auroc"])


if __name__ == "__main__":
    unittest.main()
