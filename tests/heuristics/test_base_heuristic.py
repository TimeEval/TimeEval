import unittest
from pathlib import Path

from timeeval import Algorithm
from timeeval.datasets import Dataset
from timeeval.heuristics import TimeEvalParameterHeuristic


class HeuristicImpl(TimeEvalParameterHeuristic):
    def __init__(self, test_str: str = "", test_float: float = .0):
        self.test_str = test_str
        self.test_float = test_float

    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
        pass


class TestBaseHeuristic(unittest.TestCase):
    def setUp(self) -> None:
        self.heuristic = HeuristicImpl(test_float=2.3)

    def test_name(self):
        self.assertEqual(self.heuristic.name, "HeuristicImpl")

    def test_parameter_names(self):
        self.assertSetEqual(set(self.heuristic.get_param_names()), {"test_str", "test_float"})

    def test_parameters(self):
        params = self.heuristic.parameters()
        self.assertDictEqual(params, {"test_str": "", "test_float": 2.3})

    def test_no_constructor(self):
        class _Heuristic(TimeEvalParameterHeuristic):
            def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
                pass

        self.assertListEqual(_Heuristic.get_param_names(), [])

    def test_fail_on_varargs(self):
        class _Heuristic(TimeEvalParameterHeuristic):
            def __init__(self, *args):
                pass

            def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> int:
                pass

        with self.assertRaises(RuntimeError):
            _Heuristic.get_param_names()
