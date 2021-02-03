from abc import ABC, abstractmethod
from typing import Optional

from ..data_types import AlgorithmParameter


class BaseAdapter(ABC):
    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError()

    def _preprocess_data(self, data: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return data

    def _postprocess_data(self, data: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return data

    def __call__(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        dataset = self._preprocess_data(dataset, args)
        dataset = self._call(dataset, args)
        return self._postprocess_data(dataset, args)

    def prepare(self):
        pass
