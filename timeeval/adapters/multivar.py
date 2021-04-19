import multiprocessing as mp
from enum import Enum
from typing import Callable, List

import numpy as np

from .base import Adapter
from ..data_types import AlgorithmParameter


class AggregationMethod(Enum):
    MEAN = 0
    MEDIAN = 1
    MAX = 2

    def __call__(self, data: List[np.ndarray]) -> np.ndarray:
        if self == self.MEAN:
            fn = np.mean
        elif self == self.MEDIAN:
            fn = np.median
        else:  # if self == self.MAX:
            fn = np.max

        return fn(np.stack(data, axis=1), axis=1).reshape(-1)


class MultivarAdapter(Adapter):
    def __init__(self, fn: Callable[[np.ndarray, dict], np.ndarray], aggregation: AggregationMethod = AggregationMethod.MEAN,
                 n_jobs: int = 1):
        self.fn = fn
        self.aggregation = aggregation
        self.n_jobs = n_jobs

    def _parallel_call(self, data: np.ndarray, args: dict) -> List[np.ndarray]:
        pool = mp.Pool(self.n_jobs)
        scores = pool.starmap(self.fn, [(data[:, c], args) for c in range(data.shape[1])])
        return scores

    def _call(self, dataset: AlgorithmParameter, args: dict) -> np.ndarray:
        if isinstance(dataset, np.ndarray):
            if self.n_jobs > 1:
                scores = self._parallel_call(dataset, args)
            else:
                scores = list()
                for dim in range(dataset.shape[1]):
                    scores.append(self.fn(dataset[:, dim], args))
            return self.aggregation(scores)
        else:
            raise ValueError(
                "MultivarAdapter can only handle np.ndarray as input. Make sure that `Algorithm(..., data_as_file=False)`!")
