import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.algorithms import SupervisedDeviatingFromMean
from timeeval import TimeEval, Algorithm, Datasets
from timeeval.data_types import TrainingType, InputDimensionality
from timeeval.datasets.datasets import Dataset
from timeeval.experiments import Experiment, Experiments
from timeeval.resource_constraints import ResourceConstraints
from timeeval.timeeval import Status
from timeeval.utils.metrics import Metric


class TestDatasetAndAlgorithmMatch(unittest.TestCase):

    def setUp(self) -> None:
        self.dmgr = Datasets("./tests/example_data")
        self.algorithms = [
            Algorithm(
                name="supervised_deviating_from_mean",
                main=SupervisedDeviatingFromMean(),
                training_type=TrainingType.SUPERVISED,
                data_as_file=False
            )
        ]

    def test_supervised_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], self.algorithms,
                                repetitions=1,
                                results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        np.testing.assert_array_almost_equal(results["ROC_AUC"].values, [0.810225])

    def test_mismatched_training_type(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean",
            main=SupervisedDeviatingFromMean(),
            training_type=TrainingType.SEMI_SUPERVISED,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], [algo],
                                repetitions=1,
                                results_path=Path(tmp_path),
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR.name)
        self.assertIn("training type", results.loc[0, "error_message"])
        self.assertIn("incompatible", results.loc[0, "error_message"])

    def test_mismatched_input_dimensionality(self):
        algo = Algorithm(
            name="supervised_deviating_from_mean",
            main=SupervisedDeviatingFromMean(),
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False
        )
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], [algo],
                                repetitions=1,
                                results_path=Path(tmp_path),
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR.name)
        self.assertIn("input dimensionality", results.loc[0, "error_message"])
        self.assertIn("incompatible", results.loc[0, "error_message"])

    def test_missing_training_dataset_timeeval(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-int")], self.algorithms,
                                repetitions=1,
                                results_path=Path(tmp_path),
                                skip_invalid_combinations=False)
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR.name)
        self.assertIn("training dataset", results.loc[0, "error_message"])
        self.assertIn("not found", results.loc[0, "error_message"])

    def test_missing_training_dataset_experiment(self):
        exp = Experiment(
            dataset=Dataset(
                datasetId=("test", "dataset-datetime"),
                dataset_type="synthetic",
                training_type=TrainingType.SUPERVISED,
                num_anomalies=1,
                dimensions=1,
                length=3000,
                period_size=-1
            ),
            algorithm=self.algorithms[0],
            params={},
            repetition=0,
            base_results_dir=Path("tmp_path"),
            resource_constraints=ResourceConstraints(),
            metrics=Metric.default_list(),
        )
        with self.assertRaises(ValueError) as e:
            exp._perform_training(None)
        self.assertIn("No training dataset", str(e.exception))

    def test_dont_skip_invalid_combinations(self):
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            datasets=datasets,
            algorithms=self.algorithms,
            metrics=Metric.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=False
        )
        self.assertEqual(len(exps), len(datasets) * len(self.algorithms))

    def test_skip_invalid_combinations(self):
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            datasets=datasets,
            algorithms=self.algorithms,
            metrics=Metric.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=True
        )
        self.assertEqual(len(exps), 1)
        exp = list(exps)[0]
        self.assertEqual(exp.dataset.training_type, exp.algorithm.training_type)
        self.assertEqual(exp.dataset.input_dimensionality, exp.algorithm.input_dimensionality)
