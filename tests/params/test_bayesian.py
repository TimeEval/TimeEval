import socket
import tempfile
import unittest
from pathlib import Path

import optuna
import pytest
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileOpenLock, JournalFileStorage

from timeeval import Algorithm, TrainingType
from timeeval.adapters import FunctionAdapter
from timeeval.datasets import Dataset
from timeeval.integration.optuna import OptunaLazyParams
from timeeval.metrics import RangePrAUC
from timeeval.params import BayesianParameterSearch


@pytest.mark.optuna
class TestBayesianParameterSearch(unittest.TestCase):

    def setUp(self) -> None:
        from optuna import distributions

        self.algorithm = Algorithm(
            name="TestAlgorithm", main=FunctionAdapter.identity()
        )
        self.dataset = Dataset(
            datasetId=("test", "dataset-name"),
            dataset_type="synthetic",
            training_type=TrainingType.UNSUPERVISED,
            length=50000,
            dimensions=1,
            contamination=0.05,
            min_anomaly_length=20,
            median_anomaly_length=52,
            max_anomaly_length=112,
            period_size=32,
            num_anomalies=6,
        )
        self.param_distributions = {
            "method": distributions.CategoricalDistribution(["good", "bad"]),
            "max_features": distributions.FloatDistribution(
                low=0.0, high=1.0, step=0.01
            ),
            "window_size": distributions.IntDistribution(low=1, high=1000, step=5),
        }
        self.expected_params = [
            {"method": "bad", "max_features": 0.73, "window_size": 596},
            {"method": "good", "max_features": 0.05, "window_size": 866},
        ]

    def test_update_options(self):
        from timeeval.integration.optuna import (
            OptunaConfiguration,
            OptunaStudyConfiguration,
        )
        from optuna.samplers import CmaEsSampler, TPESampler

        default_sampler = TPESampler()
        study_sampler = CmaEsSampler(with_margin=True)
        metric = RangePrAUC()
        global_config = OptunaConfiguration(
            default_storage="sqlite:///optuna.db",
            default_sampler=default_sampler,
            continue_existing_studies=True,
            dashboard=True,
        )
        optuna_study_config = OptunaStudyConfiguration(
            n_trials=4,
            sampler=study_sampler,
            metric=metric,
            direction="maximize",
        )
        self.assertEqual(optuna_study_config.n_trials, 4)
        self.assertIsNone(optuna_study_config.storage)
        self.assertEqual(optuna_study_config.sampler, study_sampler)
        self.assertIsNone(optuna_study_config.pruner)
        self.assertEqual(optuna_study_config.metric, metric)
        self.assertEqual(optuna_study_config.direction, "maximize")
        self.assertFalse(optuna_study_config.continue_existing_study)

        optuna_study_config = optuna_study_config.update_unset_options(global_config)
        self.assertEqual(optuna_study_config.n_trials, 4)
        self.assertEqual(optuna_study_config.storage, "sqlite:///optuna.db")
        self.assertEqual(optuna_study_config.sampler, study_sampler)
        self.assertIsNone(optuna_study_config.pruner)
        self.assertEqual(optuna_study_config.metric, metric)
        self.assertEqual(optuna_study_config.direction, "maximize")
        self.assertTrue(optuna_study_config.continue_existing_study)

    def test_lazy_param_creation(self):
        from timeeval.integration.optuna import OptunaStudyConfiguration

        optuna_study_config = OptunaStudyConfiguration(
            n_trials=4, metric=RangePrAUC(), storage=None
        )
        param_search = BayesianParameterSearch(
            config=optuna_study_config, params=self.param_distributions
        )
        params = list(param_search.iter(self.algorithm, self.dataset))
        self.assertEqual(len(params), 4)

        for i, param in enumerate(params):
            trial_id = f"{self.algorithm.name}-{self.dataset.name}-{i}"
            self.assertIsInstance(param, OptunaLazyParams)
            self.assertEqual(len(param), len(self.param_distributions))
            self.assertListEqual(list(param), list(self.param_distributions.keys()))
            self.assertListEqual(
                list(param.items()),
                list(map(lambda k: (k, trial_id), self.param_distributions.keys())),
            )
            self.assertEqual(param.uid(), trial_id)

    def test_materialization(self):
        from timeeval.integration.optuna import OptunaStudyConfiguration

        study_name = f"{self.algorithm.name}-{self.dataset.name}"

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            journal_file_path = str(
                tmp_path / "test_materialization.optuna-journal.log"
            )
            optuna_study_config = OptunaStudyConfiguration(
                n_trials=4,
                metric=RangePrAUC(),
                storage=lambda: JournalStorage(
                    JournalFileStorage(
                        journal_file_path,
                        lock_obj=JournalFileOpenLock(journal_file_path),
                    )
                ),
                sampler=TPESampler(seed=42),
            )
            param_search = BayesianParameterSearch(
                config=optuna_study_config, params=self.param_distributions
            )
            params = list(param_search.iter(self.algorithm, self.dataset))
            loaded_study = optuna.load_study(
                study_name, storage=optuna_study_config.storage()
            )
            self.assertEqual(loaded_study.study_name, study_name)
            self.assertEqual(loaded_study.user_attrs["algorithm"], self.algorithm.name)
            self.assertEqual(loaded_study.user_attrs["dataset"], self.dataset.name)
            self.assertEqual(
                loaded_study.user_attrs["metric"], optuna_study_config.metric.name
            )
            self.assertEqual(
                loaded_study.user_attrs["direction"],
                str(optuna_study_config.direction).lower(),
            )

            for i, (param, p_suggested) in enumerate(zip(params, self.expected_params)):
                trial_id = f"{self.algorithm.name}-{self.dataset.name}-{i}"
                self.assertIsInstance(param, OptunaLazyParams)
                param.materialize()

                self.assertIsNotNone(param._study)
                self.assertIsNotNone(param._trial)
                self.assertEqual(param.trial().user_attrs["uid"], trial_id)
                self.assertEqual(param.trial().user_attrs["node"], socket.gethostname())

                self.assertIsInstance(param["method"], str)
                self.assertIsInstance(param["max_features"], float)
                self.assertIsInstance(param["window_size"], int)
                self.assertListEqual(list(param), list(p_suggested))
                # TimeEval ignores seeding on purpose, so we can't check for the actual values
                # self.assertEqual(param["method"], p_suggested["method"])
                # self.assertEqual(param["max_features"], p_suggested["max_features"])
                # self.assertEqual(param["window_size"], p_suggested["window_size"])
                # self.assertDictEqual(param.to_dict(), p_suggested)

    def test_empty_params(self):
        from timeeval.integration.optuna import OptunaStudyConfiguration

        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            journal_file_path = str(tmp_path / "test_empty_params.optuna-journal.log")
            optuna_study_config = OptunaStudyConfiguration(
                n_trials=4,
                metric=RangePrAUC(),
                storage=lambda: JournalStorage(
                    JournalFileStorage(
                        journal_file_path,
                        lock_obj=JournalFileOpenLock(journal_file_path),
                    )
                ),
                sampler=TPESampler(seed=42),
            )
            param_search = BayesianParameterSearch(
                config=optuna_study_config, params={}
            )
            params = list(param_search.iter(self.algorithm, self.dataset))
            for p in params:
                p.materialize()
                self.assertDictEqual(p.to_dict(), {})
