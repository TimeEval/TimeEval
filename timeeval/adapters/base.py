import numpy as np
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def _call(self, dataset: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError()

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data

    def _postprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data

    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        dataset = self._preprocess_data(dataset)
        dataset = self._call(dataset)
        return self._postprocess_data(dataset)
