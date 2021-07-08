import unittest

import numpy as np

from timeeval import Algorithm, TrainingType
from timeeval.adapters import FunctionAdapter


class TestAlgorithm(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.random.rand(10)
        self.unsupervised_algorithm = Algorithm(
            name="TestAlgorithm",
            main=FunctionAdapter.identity(),
            training_type=TrainingType.UNSUPERVISED
        )
        self.supervised_algorithm = Algorithm(
            name="TestAlgorithm",
            main=FunctionAdapter.identity(),
            training_type=TrainingType.SUPERVISED
        )
        self.semi_supervised_algorithm = Algorithm(
            name="TestAlgorithm",
            main=FunctionAdapter.identity(),
            training_type=TrainingType.SEMI_SUPERVISED
        )

    def test_execution(self):
        result = self.unsupervised_algorithm.execute(self.data)
        np.testing.assert_array_equal(self.data, result)

        result = self.semi_supervised_algorithm.execute(self.data)
        np.testing.assert_array_equal(self.data, result)

        result = self.supervised_algorithm.execute(self.data)
        np.testing.assert_array_equal(self.data, result)

    def test_unsupervised_training(self):
        with self.assertRaises(ValueError) as e:
            self.unsupervised_algorithm.train(self.data)
        self.assertRegex(str(e.exception), r".*[Cc]alling.*train.*unsupervised algorithm.*not supported.*")

    def test_semi_and_supervised_training(self):
        result = self.semi_supervised_algorithm.train(self.data)
        np.testing.assert_array_equal(self.data, result)

        result = self.supervised_algorithm.train(self.data)
        np.testing.assert_array_equal(self.data, result)
