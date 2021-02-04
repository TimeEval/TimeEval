import unittest
from unittest.mock import patch
import numpy as np
import subprocess

from timeeval.adapters import MultivarAdapter, DistributedAdapter, JarAdapter
from timeeval.adapters.multivar import AggregationMethod
from timeeval.adapters.base import Adapter


def deviating_from_median(data: np.ndarray, args: dict = {}) -> np.ndarray:
    diffs = np.abs((data - np.median(data)))
    return diffs / diffs.max()


class TestMultivarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(100, 3)
        self.y = (x := np.abs(self.X - np.median(self.X, axis=0))) / x.max(axis=0)
        self.y_median = np.median(self.y, axis=1)
        self.y_mean = np.mean(self.y, axis=1)
        self.y_max = np.max(self.y, axis=1)

    def test_multivar_deviating_from_median_mean(self):
        algorithm = MultivarAdapter(deviating_from_median, AggregationMethod.MEAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_mean, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_median(self):
        algorithm = MultivarAdapter(deviating_from_median, AggregationMethod.MEDIAN)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_median, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_max(self):
        algorithm = MultivarAdapter(deviating_from_median, AggregationMethod.MAX)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_max, score)
        self.assertEqual(len(self.X), len(score))

    def test_multivar_deviating_from_median_parallel(self):
        algorithm = MultivarAdapter(deviating_from_median, AggregationMethod.MEAN, n_jobs=2)
        score = algorithm(self.X)

        np.testing.assert_array_equal(self.y_mean, score)
        self.assertEqual(len(self.X), len(score))


class SSHProcess:
    def __init__(self):
        class STDin:
            def __init__(self):
                self.written = list()

            def write(self, cmd):
                self.written.append(cmd)

            def close(self):
                pass

        self.stdin = STDin()


class TestDistributedAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.random.rand(1000, 10)
        self.remote_command = "test_command"
        self.remote_user = "test_user"
        self.remote_hosts = [
            "testhost01",
            "testhost02"
        ]

    @patch('subprocess.Popen')
    def test_screen_command(self, mock_A):
        cmd = f'screen -dm bash -c "{self.remote_command}"'

        ssh_processes = [SSHProcess(), SSHProcess()]
        mock_A.side_effect = ssh_processes

        algorithm = DistributedAdapter(deviating_from_median,
                                       remote_command=self.remote_command,
                                       remote_user=self.remote_user,
                                       remote_hosts=self.remote_hosts)

        algorithm(self.X)

        for p in range(len(ssh_processes)):
            self.assertEqual(len(ssh_processes[p].stdin.written), 1)
            self.assertEqual(ssh_processes[p].stdin.written[0], cmd)


class TestJarAdapter(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(4444)
        self.X = np.random.rand(1000, 10)

    @patch('timeeval.adapters.JarAdapter._read_results')
    @patch('subprocess.call')
    def test_args_formatting(self, mock_A, mock_B):
        mock_B.side_effect = [np.random.rand(100), np.random.rand(100)]

        jar_file = "file.jar"
        output_file = "output_file"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=False)
        algorithm(self.X)

        args, kwargs = mock_A.call_args
        self.assertListEqual(args[0], ['java', '-jar', jar_file, "plot", "test", "--plot", "yes", "--test", "2"])
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)

    @patch('timeeval.adapters.JarAdapter._read_results')
    @patch('subprocess.call')
    def test_args_formatting_verbose(self, mock_A, mock_B):
        mock_B.side_effect = [np.random.rand(100), np.random.rand(100)]

        jar_file = "file.jar"
        output_file = "output_file"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=True)
        algorithm(self.X)

        _, kwargs = mock_A.call_args
        self.assertEqual(kwargs["stdout"], subprocess.STDOUT)

    @patch('subprocess.call')
    def test_args_read_results(self, mock_A):
        output_file = "tests/example_data/data.txt"
        result = np.loadtxt(output_file)
        jar_file = "file.jar"
        args = ["plot", "test"]
        kwargs = dict(plot="yes", test=2)
        algorithm = JarAdapter(jar_file=jar_file, output_file=output_file, args=args, kwargs=kwargs, verbose=True)
        np.testing.assert_array_equal(result, algorithm(self.X))

        _, kwargs = mock_A.call_args
        self.assertEqual(kwargs["stdout"], subprocess.STDOUT)


class TestBaseAdapter(unittest.TestCase):
    def test_type_error(self):
        with self.assertRaises(TypeError):
            Adapter()
