from abc import ABC, abstractmethod
from typing import Optional, Callable

from ..data_types import AlgorithmParameter


class Adapter(ABC):

    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError("A subclass of Adapter must implement _call()!")

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        return None

    def __call__(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        return self._call(dataset, args)

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        return None
