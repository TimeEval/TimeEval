from abc import ABC, abstractmethod
from typing import Optional

from ..data_types import AlgorithmParameter


class Adapter(ABC):
    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError()

    def prepare(self):
        pass

    def __call__(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return self._call(dataset, args)

    def finalize(self):
        pass
