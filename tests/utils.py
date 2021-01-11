import unittest
import numpy as np

from timeeval.utils.window import reverse_windowing, reverse_windowing_iterative


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array([1, 2, 3, 4, 5], dtype=np.float)
        self.y_mean = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

    def test_reverse_windowing_mean(self):
        y_reversed = reverse_windowing(self.X, window_size=5)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_reverse_windowing_iterative_mean(self):
        y_reversed = reverse_windowing_iterative(self.X, window_size=5)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())


if __name__ == "__main__":
    unittest.main()
