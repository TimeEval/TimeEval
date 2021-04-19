from abc import ABC, abstractmethod
from typing import Optional, Callable

from ..data_types import AlgorithmParameter, ExecutionType


class Adapter(ABC):

    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:  # pragma: no cover
        raise NotImplementedError("A subclass of Adapter must implement _call()!")

    def __call__(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        args = args or {}
        if "executionType" not in args:
            args["executionType"] = ExecutionType.EXECUTE
        return self._call(dataset, args)

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        return None

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        return None
