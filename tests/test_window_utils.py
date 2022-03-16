import os
import unittest

import numpy as np
import pytest

from timeeval.utils.window import ReverseWindowing, Method, padding_borders


class TestReverseWindowing(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.arange(1, 201, dtype=np.float64)
        self.y_mean = np.concatenate([np.arange(1, 3.5, .5), np.arange(4, 198), np.arange(198, 200.5, .5)])
        self.y_mean_non_reversed = np.arange(3, 199)
        self.y_median = np.concatenate([np.arange(1, 3.5, .5), np.arange(4, 198), np.arange(198, 200.5, .5)])
        self.y_sum = np.concatenate(
            [np.array([1, 3, 6, 10]), np.arange(15, 199 * 5, 5), np.array([794, 597, 399, 200])]
        )

    def test_reverse_windowing_vectorized_mean(self):
        y_reversed = ReverseWindowing(window_size=5).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    def test_reverse_windowing_vectorized_median(self):
        y_reversed = ReverseWindowing(window_size=5, reduction=Method.MEDIAN).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_median, y_reversed)

    def test_reverse_windowing_vectorized_sum(self):
        y_reversed = ReverseWindowing(window_size=5, reduction=Method.SUM).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_sum, y_reversed)

    def test_reverse_windowing_iterative_mean(self):
        y_reversed = ReverseWindowing(window_size=5)._reverse_windowing_iterative(self.X, is_chunk=False)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    def test_reverse_windowing_chunks_iterative_mean(self):
        y_reversed = ReverseWindowing(window_size=5, chunksize=10)._reverse_windowing_iterative(self.X, is_chunk=False)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    def test_reverse_windowing_chunks_iterative_single_chunk_mean(self):
        y_reversed = ReverseWindowing(window_size=5, chunksize=10)._reverse_windowing_iterative(self.X, is_chunk=True)
        np.testing.assert_array_equal(self.y_mean_non_reversed, y_reversed)

    def test_reverse_windowing_chunks_vectorized_mean(self):
        y_reversed = ReverseWindowing(window_size=5, chunksize=10).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    @pytest.mark.skipif(condition=os.getenv("CI", "false") == "true", reason="CI never finishes on sopedu")
    def test_mp_windowing_iterative_mean(self):
        y_reversed = ReverseWindowing(window_size=5, n_jobs=4).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    @pytest.mark.skipif(condition=os.getenv("CI", "false") == "true", reason="CI never finishes on sopedu")
    def test_mp_windowing_chunks_mean(self):
        y_reversed = ReverseWindowing(window_size=5, n_jobs=4, chunksize=10).fit_transform(self.X)
        np.testing.assert_array_equal(self.y_mean, y_reversed)

    def test_chunksize_is_none_exception(self):
        with self.assertRaises(ValueError):
            ReverseWindowing(window_size=5)._chunk_and_vectorize(self.X)

    def test_chunk_array_without_pad_end(self):
        chunks = ReverseWindowing(window_size=5)._chunk_array(self.X, 2, True, False)
        np.testing.assert_array_equal(self.X[100:], chunks[1])

    def test_padding_borders(self):
        padded = np.concatenate([np.zeros(5), self.X, np.zeros(5)])
        y_padded = padding_borders(self.X, len(self.X) + 10)
        np.testing.assert_array_equal(padded, y_padded)


if __name__ == "__main__":
    unittest.main()
