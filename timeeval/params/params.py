from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping, Dict

from timeeval.utils.hash_dict import hash_dict


# only imports the below classes for type checking to avoid circular imports (annotations-import is necessary!)
if TYPE_CHECKING:
    import numpy as np


class Params(Mapping[str, Any], ABC):

    @abstractmethod
    def materialize(self) -> Params:
        ...

    @abstractmethod
    def assess(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        ...

    @abstractmethod
    def fail(self) -> None:
        ...

    @abstractmethod
    def uid(self) -> str:
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...


class FixedParams(Dict[str, Any], Params):
    def materialize(self) -> FixedParams:
        return self

    def assess(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return 0.0

    def fail(self) -> None:
        pass

    def uid(self) -> str:
        return hash_dict(self)

    def to_dict(self) -> Dict[str, Any]:
        return self
