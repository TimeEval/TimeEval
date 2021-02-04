import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from timeeval.adapters import JarAdapter


class TestJarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(1000, 10)

    @patch('timeeval.adapters.JarAdapter._read_results')
    @patch('subprocess.call')
    def test_args_formatting(self, call_mock, result_mock):
        result_mock.side_effect = [np.random.rand(100), np.random.rand(100)]

        jar_file = "file.jar"
        output_file = "output_file"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=False)
        algorithm(self.X)

        args, kwargs = call_mock.call_args
        self.assertListEqual(args[0], ['java', '-jar', jar_file, "plot", "test", "--plot", "yes", "--test", "2"])
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)

    @patch('timeeval.adapters.JarAdapter._read_results')
    @patch('subprocess.call')
    def test_args_formatting_verbose(self, call_mock, result_mock):
        result_mock.side_effect = [np.random.rand(100), np.random.rand(100)]

        jar_file = "file.jar"
        output_file = "output_file"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=True)
        algorithm(self.X)

        _, kwargs = call_mock.call_args
        self.assertEqual(kwargs["stdout"], subprocess.STDOUT)

    @patch('subprocess.call')
    def test_args_read_results(self, call_mock):
        output_file = "tests/example_data/data.txt"
        result = np.loadtxt(output_file)
        jar_file = "file.jar"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=True)
        np.testing.assert_array_equal(result, algorithm(self.X))

        _, kwargs = call_mock.call_args
        self.assertEqual(kwargs["stdout"], subprocess.STDOUT)
