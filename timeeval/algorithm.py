from dataclasses import dataclass
from typing import Optional
from sklearn.model_selection import ParameterGrid

from .data_types import TSFunction, TSFunctionPost


@dataclass
class Algorithm:
    name: str
    main: TSFunction
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
    param_grid: ParameterGrid = ParameterGrid({})
