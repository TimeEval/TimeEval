from abc import ABC, abstractmethod
from typing import Optional

from ..data_types import AlgorithmParameter


class BaseAdapter(ABC):
    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError()

    def __call__(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return self._call(dataset, args)

    def prepare(self):
        pass
