import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class BaseAdapter(ABC):
    @abstractmethod
    def _call(self, dataset: Union[np.ndarray, Path]) -> Union[np.ndarray, Path]:  # pragma: no cover
        raise NotImplementedError()

    def _preprocess_data(self, data: Union[np.ndarray, Path]) -> Union[np.ndarray, Path]:
        return data

    def _postprocess_data(self, data: Union[np.ndarray, Path]) -> np.ndarray:
        return data

    def __call__(self, dataset: Union[np.ndarray, Path]) -> np.ndarray:
        dataset = self._preprocess_data(dataset)
        dataset = self._call(dataset)
        return self._postprocess_data(dataset)
