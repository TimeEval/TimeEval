import unittest
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from tests.fixtures.algorithms import SupervisedDeviatingFromMean
from timeeval import Datasets, Algorithm, TrainingType, InputDimensionality, Metric
from timeeval.experiments import Experiments


class TestUniqueExperimentName(unittest.TestCase):

    def setUp(self) -> None:
        self.dmgr = Datasets("./tests/example_data")
        self.algorithms = [
            Algorithm(
                name="supervised_deviating_from_mean",
                main=SupervisedDeviatingFromMean(),
                training_type=TrainingType.SUPERVISED,
                input_dimensionality=InputDimensionality.UNIVARIATE,
                data_as_file=False,
                param_grid=ParameterGrid({
                    "param1": range(3),
                    "param2": ["a", "b"]
                })
            )
        ]

    def test_unique_experiment_name(self):
        """
        The experiment names are used as keys for the distributed evaluation tasks and must be unique!
        """
        datasets = [self.dmgr.get(d) for d in self.dmgr.select()]
        exps = Experiments(
            dmgr=self.dmgr,
            datasets=datasets,
            algorithms=self.algorithms,
            repetitions=2,
            metrics=Metric.default_list(),
            base_result_path=Path("tmp_path"),
            skip_invalid_combinations=True
        )
        self.assertEqual(len(exps), 12)
        exp_names = [exp.name for exp in exps]
        self.assertEqual(len(exp_names), len(set(exp_names)))
