import unittest

import numpy as np

import tests.fixtures.heuristics_fixtures as fixtures
from timeeval.heuristics import TimeEvalHeuristic, TimeEvalParameterHeuristic, inject_heuristic_values


class TestInjection(unittest.TestCase):
    def test_successful_factory(self):
        obj = TimeEvalHeuristic("AnomalyLengthHeuristic(agg_type='median')")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("CleanStartSequenceSizeHeuristic(max_factor=0.1)")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("RelativeDatasetSizeHeuristic(factor=0.2)")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("ParameterDependenceHeuristic(source_parameter='x', factor=0.2)")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("PeriodSizeHeuristic()")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("EmbedDimRangeHeuristic()")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("ContaminationHeuristic()")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("DefaultFactorHeuristic(factor=1.234)")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("DefaultExponentialFactorHeuristic(exponent=2)")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)
        obj = TimeEvalHeuristic("DatasetIdHeuristic()")
        self.assertIsInstance(obj, TimeEvalParameterHeuristic)

    def test_unknown_heuristic_factory(self):
        with self.assertRaises(ValueError):
            TimeEvalHeuristic("bla")
        with self.assertRaises(ValueError):
            TimeEvalHeuristic("1+1 == 2")
        with self.assertRaises(ValueError):
            TimeEvalHeuristic("Algorithm(num_anomalies=3)")

    def test_wrong_parameter_factory(self):
        with self.assertRaises(TypeError) as e:
            TimeEvalHeuristic("RelativeDatasetSizeHeuristic(dummy=0)")
        self.assertIn("unexpected keyword argument", str(e.exception))

    def test_inject_heuristic_values(self):
        params = {
            "x": 2,
            "window_size": "heuristic:AnomalyLengthHeuristic(agg_type='median')",
            "n_init": "heuristic:CleanStartSequenceSizeHeuristic(max_factor=0.75)",
            "size": "heuristic:RelativeDatasetSizeHeuristic(factor=0.1)",
            "y": "heuristic:ParameterDependenceHeuristic(source_parameter='x', fn=lambda x: str(x)+'%')",
            "period": "heuristic:PeriodSizeHeuristic()",
            "embed_dim_range": "heuristic:EmbedDimRangeHeuristic(base_factor=2)",
            "to_be_removed": "heuristic:ParameterDependenceHeuristic(source_parameter='missing')",
            "contamination": "heuristic:ContaminationHeuristic()",
            "alpha": "heuristic:DefaultFactorHeuristic(factor=2.0)",
            "beta": "heuristic:DefaultExponentialFactorHeuristic(exponent=-2)",
            "dataset_id": "heuristic:DatasetIdHeuristic()"
        }
        # required by DefaultFactorHeuristics and DefaultExponentialFactorHeuristic:
        fixtures.algorithm.param_schema = {"alpha": {"defaultValue": 0.1}, "beta": {"defaultValue": 5.0}}
        new_params = inject_heuristic_values(
            params=params,
            algorithm=fixtures.algorithm,
            dataset_details=fixtures.dataset,
            dataset_path=fixtures.real_test_dataset_path
        )
        expected_embed_dim_range = (
            [50, 100, 150] if fixtures.dataset.period_size is None else
            (np.array([0.5, 1.0, 1.5]) * fixtures.dataset.period_size * 2).astype(int)
        )
        actual_embed_dim_range = new_params["embed_dim_range"]
        del new_params["embed_dim_range"]

        self.assertDictEqual(new_params, {
            "x": 2,
            "window_size": fixtures.dataset.median_anomaly_length,
            "n_init": 2250,
            "size": 300,
            "y": "2%",
            "period": 1 if fixtures.dataset.period_size is None else fixtures.dataset.period_size,
            "contamination": 1.0 / 3600,
            "alpha": 0.2,
            "beta": 0.05,
            "dataset_id": fixtures.dataset.datasetId
        })
        np.testing.assert_array_equal(actual_embed_dim_range, expected_embed_dim_range)
