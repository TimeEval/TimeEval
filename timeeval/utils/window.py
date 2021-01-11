import numpy as np
import tqdm
from enum import Enum


class Method(Enum):
    MEAN = 0
    MEDIAN = 1
    SUM = 2

    def fn(self, x: np.ndarray) -> float:
        if self == self.MEAN:
            return np.nanmean(x).item()
        elif self == self.MEDIAN:
            return np.nanmedian(x).item()
        elif self == self.SUM:
            return np.nansum(x).item()
        else:
            raise ValueError("Reduction method isn't supported!")


def reverse_windowing(scores: np.ndarray, window_size: int, reduction: Method = Method.MEAN) -> np.ndarray:
    unwindowed_length = (window_size - 1) + len(scores)
    mapped = np.zeros((unwindowed_length, window_size))
    for p, s in enumerate(scores):
        assignment = np.eye(window_size) * s
        mapped[p:p + window_size] += assignment
    if reduction == Method.MEAN:
        # transform non-score zeros to NaNs
        h = np.tril([1.] * window_size, 0)
        h[h == 0] = np.nan
        h2 = np.triu([1.] * window_size, 0)
        h2[h2 == 0] = np.nan
        mapped[:window_size] *= h
        mapped[-window_size:] *= h2
        mapped = np.nanmean(mapped, axis=1)
    elif reduction == Method.SUM:
        mapped = mapped.sum(axis=1)
    return mapped


def reverse_windowing_iterative(scores: np.ndarray, window_size: int, reduction: Method = Method.MEAN) -> np.ndarray:
    scores = np.pad(scores, (window_size - 1, window_size - 1), 'constant', constant_values=(np.nan, np.nan))

    for i in tqdm.trange(len(scores) - (window_size - 1), desc="reverse windowing"):
        scores[i] = reduction.fn(scores[i:i + window_size])

    return scores[~np.isnan(scores)]


def padding_borders(scores: np.ndarray, input_size: int) -> np.ndarray:
    padding_size = (input_size - len(scores)) // 2
    result = np.zeros(input_size)
    result[padding_size:padding_size+len(scores)] = scores
    return result
