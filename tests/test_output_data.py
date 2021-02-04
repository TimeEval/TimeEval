import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from freezegun import freeze_time
from sklearn.model_selection import ParameterGrid

from timeeval import TimeEval, Algorithm, Datasets
from timeeval.adapters import FunctionAdapter
from timeeval.timeeval import AlgorithmParameter, ANOMALY_SCORES_TS, METRICS_CSV, EXECUTION_LOG, HYPER_PARAMETERS, RESULTS_CSV
from timeeval.utils.hash_dict import hash_dict


def deviating_from_mean(X: AlgorithmParameter, args: dict):
    # Keep this print statement! It is captured in the execution log!
    print(args.get("results_path", TimeEval.DEFAULT_RESULT_PATH))
    diffs = np.abs((X - np.mean(X)))  # type: ignore
    diffs = diffs / diffs.max()
    return diffs


def deviating_from_mean_own_scores(X: AlgorithmParameter, args: dict):
    diffs = deviating_from_mean(X, args)
    result_path = args.get("results_path", TimeEval.DEFAULT_RESULT_PATH)
    result_path.mkdir(parents=True, exist_ok=True)  # FIXME: should be handled by TimeEval --> remove
    np.zeros_like(diffs).tofile(result_path / ANOMALY_SCORES_TS, sep="\n")
    return diffs


@freeze_time("2021-01-01")
class TestOutputData(unittest.TestCase):
    def setUp(self) -> None:
        self.DATASET = ("test", "dataset-int")
        self.datasets = Datasets("./tests/example_data", custom_datasets_file=Path("./tests/example_data/datasets.json"))
        self.hyper_params = ParameterGrid({"a": [0]})
        self.hash = hash_dict(self.hyper_params[0])
        self.algorithms = [
            Algorithm(name="deviating_from_mean", main=FunctionAdapter(deviating_from_mean), param_grid=self.hyper_params)
        ]

    def test_output_files_exists(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            timeeval = TimeEval(self.datasets, [self.DATASET], self.algorithms, results_path=tmp_path)
            timeeval.run()
            parent_path = tmp_path / "2021_01_01_00_00_00" / "deviating_from_mean" / self.hash / "test" / "dataset-int" / "1"

            self.assertTrue((parent_path / ANOMALY_SCORES_TS).exists())
            self.assertTrue((parent_path / EXECUTION_LOG).exists())
            self.assertTrue((parent_path / METRICS_CSV).exists())
            self.assertTrue((parent_path / HYPER_PARAMETERS).exists())
            self.assertTrue((tmp_path / "2021_01_01_00_00_00" / RESULTS_CSV).exists())

    def test_log_exists_and_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            timeeval = TimeEval(self.datasets, [self.DATASET], self.algorithms, results_path=tmp_path)
            timeeval.run()
            parent_path = tmp_path / "2021_01_01_00_00_00" / "deviating_from_mean" / self.hash / "test" / "dataset-int" / "1"

            self.assertEqual(str(parent_path)+"\n", (parent_path / EXECUTION_LOG).open("r").read())

    def test_hyper_params_json_exists_and_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            timeeval = TimeEval(self.datasets, [self.DATASET], self.algorithms, results_path=tmp_path)
            timeeval.run()
            parent_path = tmp_path / "2021_01_01_00_00_00" / "deviating_from_mean" / self.hash / "test" / "dataset-int" / "1"

            with (parent_path / HYPER_PARAMETERS).open("r") as f:
                self.assertEqual(f.read(), '{"a": 0}')

    def test_results_csv_exists_and_is_correct(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], self.algorithms, results_path=tmp_path)
            timeeval.run()

            true_results = pd.read_csv("./tests/example_data/results.csv")
            results = pd.read_csv(tmp_path / "2021_01_01_00_00_00" / RESULTS_CSV)

            self.assertEqual(true_results.iloc[0, 3], results.iloc[0, 3])
            self.assertEqual(results.hyper_params.values[0], '{"a": 0}')
            self.assertEqual(str(results.hyper_params_id.values[0]), hash_dict({"a": 0}))
