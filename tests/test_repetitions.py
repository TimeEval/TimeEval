import tempfile
import unittest
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd

from tests.fixtures.algorithms import DeviatingFromMean, DeviatingFromMedian, ErroneousAlgorithm
from timeeval import TimeEval, Algorithm, DatasetManager


class TestRepetitions(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=DeviatingFromMean()),
            Algorithm(name="deviating_from_median", main=DeviatingFromMedian())
        ]

    def test_multiple_results(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results()

        np.testing.assert_array_almost_equal(results["ROC_AUC_mean"].values, self.results["ROC_AUC"].values)
        self.assertEqual(results["repetitions"].unique()[0], 3)

    def test_return_no_aggregation(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)
        self.assertEqual(len(timeeval.exps.datasets) * len(self.algorithms) * 3, len(results))

    def test_return_aggregated_but_not_short(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=True, short=False)
        self.assertIn("ROC_AUC_mean", results.columns)
        self.assertIn("ROC_AUC_std", results.columns)
        np.testing.assert_array_almost_equal(results["ROC_AUC_mean"].values, self.results["ROC_AUC"].values)
        self.assertEqual(len(timeeval.exps.datasets) * len(self.algorithms), len(results))

    def test_error_in_repetition(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)

        algorithms = [
            # two datasets with 3 repititions each: skip only one repetition of the second dataset: 5
            Algorithm(name="deviating_from_mean", main=ErroneousAlgorithm(raise_after_n_calls=5)),
            Algorithm(name="deviating_from_median", main=DeviatingFromMedian())
        ]
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results()

        np.testing.assert_array_almost_equal(results["ROC_AUC_mean"].values, self.results["ROC_AUC"].values)
        # first algorithm performs 5 reps (misses one on the second dataset), second algorithm performs all 6 reps
        self.assertListEqual(results["repetitions"].tolist(), [3, 2, 3, 3])
