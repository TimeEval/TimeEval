import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.fixtures.algorithms import DeviatingFromMean, DeviatingFromMedian
from tests.fixtures.dataset_fixtures import CUSTOM_DATASET_PATH
from timeeval import TimeEval, Algorithm, AlgorithmParameter, DatasetManager, InputDimensionality


def generates_results(dataset, from_file: bool = False) -> pd.DataFrame:
    def preprocess(x: AlgorithmParameter, args: dict) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        else:  # if isinstance(x, (PosixPath, WindowsPath)):
            return pd.read_csv(x).values[:, 1:-1]

    algorithms = [
        Algorithm(name="deviating_from_mean", main=DeviatingFromMean(), preprocess=preprocess, data_as_file=from_file),
        Algorithm(name="deviating_from_median", main=DeviatingFromMedian(), preprocess=preprocess, data_as_file=from_file)
    ]

    datasets = DatasetManager("./tests/example_data", custom_datasets_file=CUSTOM_DATASET_PATH)

    with tempfile.TemporaryDirectory() as tmp_path:
        timeeval = TimeEval(datasets, [dataset], algorithms, results_path=Path(tmp_path))
        timeeval.run()
    return timeeval.results


def generates_results_multi(dataset) -> pd.DataFrame:
    algorithms = [
        Algorithm(name="deviating_from_mean",
                  main=DeviatingFromMean(),
                  data_as_file=False,
                  input_dimensionality=InputDimensionality.MULTIVARIATE),
        Algorithm(name="deviating_from_median",
                  main=DeviatingFromMedian(),
                  data_as_file=False,
                  input_dimensionality=InputDimensionality.MULTIVARIATE)
    ]

    datasets = DatasetManager("./tests/example_data", custom_datasets_file=CUSTOM_DATASET_PATH)
    with tempfile.TemporaryDirectory() as tmp_file:
        timeeval = TimeEval(datasets, [dataset], algorithms, results_path=Path(tmp_file))
        timeeval.run()
    return timeeval.results


class TestImportData(unittest.TestCase):
    def setUp(self) -> None:
        #  We only compare the columns "algorithm", "collection", "dataset", "score"
        #  without the time measurements, status and error messages
        #  (columns: "preprocessing_time", "main_time", "postprocessing_time", "status", "error_messages").
        self.results = pd.read_csv("./tests/example_data/results.csv")
        self.multi_results = pd.read_csv("./tests/example_data/results_multi.csv")
        self.KEYS = ['algorithm', 'collection', 'dataset', 'ROC_AUC']

    def test_generates_correct_results(self):
        DATASET = ("custom", "dataset.1")
        generated_results = generates_results(DATASET)
        true_results = self.results[self.results.dataset == DATASET[1]]

        np.testing.assert_array_equal(generated_results[self.KEYS].values, true_results[self.KEYS].values)

    def test_generates_correct_results_from_multi_file(self):
        DATASET = ("custom", "dataset.4")
        generated_results = generates_results_multi(DATASET)
        true_results = self.multi_results[self.multi_results.dataset == DATASET[1]]

        np.testing.assert_array_equal(generated_results[self.KEYS].values, true_results[self.KEYS].values)

    def test_algorithm_with_filename(self):
        DATASET = ("custom", "dataset.1")
        generated_results = generates_results(DATASET, from_file=True)
        true_results = self.results[self.results.dataset == DATASET[1]]

        np.testing.assert_array_equal(generated_results[self.KEYS].values, true_results[self.KEYS].values)


if __name__ == "__main__":
    unittest.main()
