from dataclasses import dataclass
from typing import Optional
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

    def prepare(self):
        self.main.prepare()

    def finalize(self):
        self.main.finalize()
