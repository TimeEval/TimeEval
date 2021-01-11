import unittest
import numpy as np

from timeeval.utils.window import reverse_windowing, reverse_windowing_iterative, reverse_windowing_iterative_parallel


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.arange(1, 201, dtype=np.float)
        self.y_mean = np.concatenate([np.arange(1, 3.5, .5), np.arange(4, 198), np.arange(198, 200.5, .5)])

    def test_reverse_windowing_mean(self):
        y_reversed = reverse_windowing(self.X, window_size=5)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_reverse_windowing_iterative_mean(self):
        y_reversed = reverse_windowing_iterative(self.X, window_size=5)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_mp_windowing_iterative_mean(self):
        y_reversed = reverse_windowing_iterative_parallel(self.X, window_size=5)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())


if __name__ == "__main__":
    unittest.main()
