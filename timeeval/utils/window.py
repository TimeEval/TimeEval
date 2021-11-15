import multiprocessing as mp
from enum import Enum
from itertools import cycle
from typing import Optional, List

import numpy as np
from sklearn.base import TransformerMixin


class Method(Enum):
    MEAN = 0
    MEDIAN = 1
    SUM = 2

    def fn(self, x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        if self == self.MEAN:
            return np.nanmean(x, axis=axis)
        elif self == self.MEDIAN:
            return np.nanmedian(x, axis=axis)
        elif self == self.SUM:
            return np.nansum(x, axis=axis)
        else:
            raise ValueError("Reduction method isn't supported!")


class ReverseWindowing(TransformerMixin):
    def __init__(self,
                 window_size: int,
                 reduction: Method = Method.MEAN,
                 n_jobs: int = 1,
                 chunksize: Optional[int] = None):
        self.window_size = window_size
        self.reduction = reduction
        self.n_jobs = n_jobs
        self.chunksize = chunksize

    def _reverse_windowing_vectorized_entire(self, scores: np.ndarray) -> np.ndarray:
        unwindowed_length = (self.window_size - 1) + len(scores)
        mapped = np.full(shape=(unwindowed_length, self.window_size), fill_value=np.nan)
        mapped[:len(scores), 0] = scores

        for w in range(1, self.window_size):
            mapped[:, w] = np.roll(mapped[:, 0], w)

        return self.reduction.fn(mapped, axis=1)

    def _reverse_windowing_vectorized_chunk(self, scores: np.ndarray) -> np.ndarray:
        mapped = np.zeros((len(scores), self.window_size)) + np.nan

        for w in range(self.window_size):
            mapped[:, w] = np.roll(scores, -w)
        mapped = mapped[:-(self.window_size-1)]

        return self.reduction.fn(mapped, axis=1)

    def _reverse_windowing_iterative(self, scores: np.ndarray, is_chunk: bool = False) -> np.ndarray:
        if not is_chunk:
            pad_n = (self.window_size - 1, self.window_size - 1)
            scores = np.pad(scores, pad_n, 'constant', constant_values=(np.nan, np.nan))

        for i in range(len(scores) - (self.window_size - 1)):
            scores[i] = self.reduction.fn(scores[i:i + self.window_size]).item()

        if is_chunk:
            scores = scores[:-(self.window_size - 1)]

        return scores[~np.isnan(scores)]

    def _reverse_windowing_parallel(self, scores: np.ndarray) -> np.ndarray:
        pool = mp.Pool(self.n_jobs)

        scores_split = self._chunk_array(scores, self.n_jobs)

        if self.chunksize is None:
            windowed_scores_split = pool.starmap(self._reverse_windowing_iterative, zip(scores_split, cycle([True])))
        else:
            windowed_scores_split = pool.starmap(self._chunk_and_vectorize, zip(scores_split, cycle([False]), cycle([False])))

        return np.concatenate(windowed_scores_split)

    def _chunk_array(self, X: np.ndarray, n_chunks: int, pad_start: bool = True, pad_end: bool = True) -> List[np.ndarray]:
        chunks = []
        len_single = len(X) // n_chunks
        for i in range(n_chunks):
            if i == 0 and pad_start:
                chunk = np.pad(X[:len_single + self.window_size - 1],
                               (self.window_size - 1, 0),
                               'constant',
                               constant_values=(np.nan, np.nan))
            elif i < (n_chunks - 1):
                chunk = X[i * len_single:(i + 1) * len_single + self.window_size - 1]
            else:
                if pad_end:
                    chunk = np.pad(X[i * len_single:],
                                   (0, self.window_size - 1),
                                   'constant',
                                   constant_values=(np.nan, np.nan))
                else:
                    chunk = X[i * len_single:]
            chunks.append(chunk)
        return chunks

    def _vectorize_chunks(self, chunks: List[np.ndarray]):
        chunked_scores = []

        for chunk in chunks:
            chunked_scores.append(self._reverse_windowing_vectorized_chunk(chunk))

        return np.concatenate(chunked_scores)

    def _chunk_and_vectorize(self, scores: np.ndarray, pad_start: bool = True, pad_end: bool = True) -> np.ndarray:
        if self.chunksize is None:
            raise ValueError("Please set an int value for chunksize in order to use _chunk_and_vectorize")
        else:
            chunks = self._chunk_array(scores, len(scores) // self.chunksize, pad_start=pad_start, pad_end=pad_end)
            return self._vectorize_chunks(chunks)

    def _has_enough_memory_for_vectorized_entire(self, n: int) -> bool:
        import sys
        import psutil
        from pathlib import Path

        container_limit_file = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        memory_limit = psutil.virtual_memory().available
        if container_limit_file.exists():
            with container_limit_file.open("r") as fh:
                container_limit = int(fh.read())
            if container_limit < 1 * 1024 ** 4:
                memory_limit = container_limit

        estimated_mem_usage = sys.getsizeof(float()) * (n + self.window_size - 1) * self.window_size
        return estimated_mem_usage <= memory_limit

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        if self.n_jobs > 1:
            return self._reverse_windowing_parallel(X)
        elif self.chunksize is not None:
            return self._chunk_and_vectorize(X)
        elif self._has_enough_memory_for_vectorized_entire(len(X)):
            return self._reverse_windowing_vectorized_entire(X)
        else:
            print("Falling back to iterative reverse windowing function to limit memory usage!")
            return self._reverse_windowing_iterative(X)


def padding_borders(scores: np.ndarray, input_size: int) -> np.ndarray:
    padding_size = (input_size - len(scores)) // 2
    result = np.zeros(input_size)
    result[padding_size:padding_size+len(scores)] = scores
    return result
