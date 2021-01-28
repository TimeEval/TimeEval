from dataclasses import dataclass
from typing import Optional

from .data_types import TSFunction, TSFunctionPost


@dataclass
class Algorithm:
    name: str
    main: TSFunction
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
