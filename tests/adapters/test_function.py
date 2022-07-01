import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

from timeeval import ResourceConstraints
from timeeval.adapters import FunctionAdapter
from timeeval.data_types import ExecutionType


class TestFunctionAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(1000, 10)
        self.captured_params: Any = None

    def _func(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        self.captured_params = params
        return data

    def test_parameter_handling(self):
        algorithm = FunctionAdapter(self._func)
        args = {
            "executionType": ExecutionType.EXECUTE,
            "resource_constraints": ResourceConstraints(),
            "hyper_params": {"param1": True, "param2": 20},
            "results_path": Path("some_path")
        }
        result = algorithm(self.X, args)
        np.testing.assert_array_equal(result, self.X)
        self.assertDictEqual(self.captured_params, args["hyper_params"])

    def test_execute(self):
        algorithm = FunctionAdapter(lambda x, _: x)
        args = {
            "executionType": ExecutionType.EXECUTE
        }
        result = algorithm(self.X, args)
        np.testing.assert_array_equal(result, self.X)
        self.assertIsNone(self.captured_params)

    def test_train(self):
        algorithm = FunctionAdapter(lambda x, _: x)
        args = {
            "executionType": ExecutionType.TRAIN
        }
        algorithm(self.X, args)
        self.assertIsNone(self.captured_params)
