import unittest

import numpy as np

from timeeval.utils.window import ReverseWindowing


class TestReverseWindowing(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.arange(1, 201, dtype=np.float64)
        self.y_mean = np.concatenate([np.arange(1, 3.5, .5), np.arange(4, 198), np.arange(198, 200.5, .5)])

    def test_reverse_windowing_vectorized_mean(self):
        y_reversed = ReverseWindowing(window_size=5).fit_transform(self.X)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_reverse_windowing_iterative_mean(self):
        y_reversed = ReverseWindowing(window_size=5)._reverse_windowing_iterative(self.X, is_chunk=False)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_reverse_windowing_chunks_mean(self):
        y_reversed = ReverseWindowing(window_size=5, chunksize=10)._reverse_windowing_iterative(self.X, is_chunk=False)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_mp_windowing_iterative_mean(self):
        y_reversed = ReverseWindowing(window_size=5, n_jobs=4).fit_transform(self.X)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())

    def test_mp_windowing_chunks_mean(self):
        y_reversed = ReverseWindowing(window_size=5, n_jobs=4, chunksize=10).fit_transform(self.X)
        self.assertListEqual(self.y_mean.tolist(), y_reversed.tolist())


if __name__ == "__main__":
    unittest.main()
