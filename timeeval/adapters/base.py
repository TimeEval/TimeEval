import numpy as np
from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def __call__(self, dataset: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data

    def _postprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data
