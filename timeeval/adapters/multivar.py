import numpy as np
from enum import Enum
from typing import Callable, List, Optional
import multiprocessing as mp
from .base import BaseAdapter, AlgorithmParameter


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

        return fn(np.stack(data, axis=1), axis=1)


class MultivarAdapter(BaseAdapter):
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], aggregation: AggregationMethod = AggregationMethod.MEAN, n_jobs: int = 1):
        self.fn = fn
        self.aggregation = aggregation
        self.n_jobs = n_jobs

    def _parallel_call(self, data: np.ndarray) -> List[np.ndarray]:
        pool = mp.Pool(self.n_jobs)
        scores = pool.map(self.fn, [data[:, c] for c in range(data.shape[1])])
        return scores

    def _call(self, dataset: np.ndarray, args: Optional[dict] = None) -> np.ndarray:
        if self.n_jobs > 1:
            scores = self._parallel_call(dataset)
        else:
            scores = list()
            for dim in range(dataset.shape[1]):
                scores.append(self.fn(dataset[:, dim]))
        return self.aggregation(scores)
