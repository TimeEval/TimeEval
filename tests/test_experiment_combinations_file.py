import tempfile
import unittest
from itertools import cycle
from pathlib import Path

import pandas as pd

from tests.fixtures.algorithms import DeviatingFromMean, DeviatingFromMedian
from timeeval import TimeEval, Algorithm, DatasetManager


class TestExperimentCombinations(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.read_csv("tests/example_data/results.csv")
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=DeviatingFromMean()),
            Algorithm(name="deviating_from_median", main=DeviatingFromMedian())
        ]

    def test_only_specific_combination(self):
        datasets_config = Path("./tests/example_data/datasets.json")
        datasets = DatasetManager("./tests/example_data", custom_datasets_file=datasets_config)

        with tempfile.TemporaryDirectory() as tmp_path:
            combinations_file = Path(tmp_path) / "combinations.csv"

            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path))
            timeeval.run()
            results = timeeval.get_results(aggregated=False)
            results = results.reset_index()[["algorithm", "collection", "dataset", "hyper_params_id"]]
            results = results[(results.algorithm == "deviating_from_mean") & (results.dataset == "dataset.3")]
            results.to_csv(combinations_file)

            timeeval = TimeEval(datasets, list(zip(cycle(["custom"]), self.results.dataset.unique())), self.algorithms,
                                repetitions=3, results_path=Path(tmp_path), experiment_combinations_file=combinations_file)
            timeeval.run()
            results = timeeval.get_results().reset_index()

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0].algorithm, "deviating_from_mean")
        self.assertEqual(results.iloc[0].dataset, "dataset.3")
