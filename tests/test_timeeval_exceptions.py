import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, List
from asyncio import Future
from itertools import cycle

from timeeval import TimeEval, Algorithm, Datasets


class TestTimeEvalExceptions(unittest.TestCase):
    def setUp(self) -> None:
        self.datasets_config = Path("./tests/example_data/datasets.json")
        self.datasets = Datasets("./tests/example_data", custom_datasets_file=self.datasets_config)

    def test_algorithm_with_file_raises_exception(self):
        file_algorithm = Algorithm(name="with_file", function=lambda x: x, data_as_file=True)

        with self.assertRaises(NotImplementedError):
            timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], [file_algorithm])
            timeeval.run()

    def test_wrong_df_shape(self):
        algorithm = Algorithm(name="test", function=lambda x: x, data_as_file=False)
        df = pd.DataFrame(np.random.rand(10, 2))
        with self.assertRaises(ValueError):
            TimeEval.evaluate(algorithm, df, ("custom", "test"))

    def test_no_algorithms(self):
        with self.assertRaises(AssertionError):
            timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], [])
            timeeval.run()

    def test_evaluation_continues_after_exception_in_algorithm(self):
        def exception_algorithm(_x):
            raise ValueError()

        algorithms = [
            Algorithm(name="exception", function=exception_algorithm, data_as_file=False),
            Algorithm(name="test", function=lambda x: x, data_as_file=False)
        ]

        timeeval = TimeEval(self.datasets, [("custom", "dataset.1")], algorithms)
        timeeval.run()

        r = timeeval.results

        self.assertEqual(r.shape[0], 1)
        self.assertEqual(r[r.algorithm == "exception"].shape[0], 0)
        self.assertEqual(r[r.algorithm == "test"].shape[0], 1)
