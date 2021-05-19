from typing import Callable, Optional

import numpy as np

from timeeval.adapters.base import Adapter
from timeeval.data_types import AlgorithmParameter, ExecutionType


def deviating_from_1d(data: np.ndarray, fn: Callable) -> np.ndarray:
    diffs = np.abs((data - fn(data)))
    return diffs / diffs.max()


def deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
    if len(data.shape) == 1:
        return deviating_from_1d(data, fn)

    diffs = np.abs((data - fn(data, axis=0)))
    diffs = diffs / diffs.max(axis=0)
    return diffs.mean(axis=1)


class DeviatingFromMean(Adapter):

    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return deviating_from(dataset, np.mean)  # type: ignore


class DeviatingFromMedian(Adapter):

    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return deviating_from(dataset, np.median)  # type: ignore


class SupervisedDeviatingFromMean(Adapter):

    def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:
        model_path = args["results_path"] / "model.txt"
        if args["executionType"] == ExecutionType.TRAIN:
            mean: np.float64 = np.mean(dataset[:, 1])  # type: ignore
            with open(model_path, "w") as f:
                f.write(str(mean))
            return dataset
        else:
            with open(model_path, "r") as f:
                mean = np.float64(f.read())

            def fn(_, **kwargs):
                return mean

            return deviating_from(dataset, fn)  # type: ignore


class ErroneousAlgorithm(Adapter):
    def __init__(self, raise_after_n_calls=0, error_message="test error"):
        self.fn = DeviatingFromMean()
        self.count = raise_after_n_calls
        self.msg = error_message

    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        if self.count <= 0:
            raise ValueError(self.msg)
        else:
            self.count -= 1
            return self.fn(dataset, args)
