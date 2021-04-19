import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures.algorithms import SupervisedDeviatingFromMean
from timeeval import TimeEval, Algorithm, Datasets
from timeeval.algorithm import TrainingType
from timeeval.experiments import Experiment
from timeeval.resource_constraints import ResourceConstraints
from timeeval.timeeval import Status
from timeeval.utils.metrics import Metric


class TestRepetitions(unittest.TestCase):

    def setUp(self) -> None:
        self.dmgr = Datasets("./tests/example_data")
        self.algorithms = [
            Algorithm(name="supervised_deviating_from_mean", main=SupervisedDeviatingFromMean(),
                      train_type=TrainingType.SUPERVISED, data_as_file=False)
        ]

    def test_supervised_algorithm(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-datetime")], self.algorithms, repetitions=1, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        np.testing.assert_array_almost_equal(results["ROC_AUC"].values, [0.810225])

    def test_missing_training_dataset_timeeval(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            timeeval = TimeEval(self.dmgr, [("test", "dataset-int")], self.algorithms, repetitions=1, results_path=Path(tmp_path))
            timeeval.run()
        results = timeeval.get_results(aggregated=False)

        self.assertEqual(results.loc[0, "status"], Status.ERROR.name)
        self.assertIn("training dataset", results.loc[0, "error_message"])
        self.assertIn("not found", results.loc[0, "error_message"])

    def test_missing_training_dataset_experiment(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            exp = Experiment(
                dataset=("test", "dataset-datetime"),
                algorithm=self.algorithms[0],
                params={},
                repetition=0,
                base_results_dir=Path(tmp_path),
                resource_constraints=ResourceConstraints(),
                metrics=Metric.default(),
            )
            with self.assertRaises(ValueError) as e:
                exp._perform_training(None)
                self.assertIn("No training dataset", str(e))
