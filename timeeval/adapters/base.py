import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path

from ..timeeval import AlgorithmParameter


class BaseAdapter(ABC):
    @abstractmethod
    def _call(self, dataset: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError()

    def _preprocess_data(self, data: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        return data

    def _postprocess_data(self, data: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        return data

    def __call__(self, dataset: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        dataset = self._preprocess_data(dataset, args)
        dataset = self._call(dataset, args)
        return self._postprocess_data(dataset, args)

    def make_available(self):
        pass
