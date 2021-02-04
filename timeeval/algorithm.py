from dataclasses import dataclass
from typing import Optional, Callable

from sklearn.model_selection import ParameterGrid

from .adapters.base import Adapter
from .data_types import TSFunction, TSFunctionPost


@dataclass
class Algorithm:
    name: str
    main: Adapter
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
    param_grid: ParameterGrid = ParameterGrid({})

    def prepare_fn(self) -> Optional[Callable[[], None]]:
        return self.main.get_prepare_fn()

    def prepare(self) -> None:
        if fn := self.prepare_fn():
            fn()

    def finalize_fn(self) -> Optional[Callable[[], None]]:
        return self.main.get_finalize_fn()

    def finalize(self) -> None:
        if fn := self.finalize_fn():
            fn()
