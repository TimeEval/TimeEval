from dataclasses import dataclass
from typing import Optional

from .adapters.base import Adapter
from .data_types import TSFunction, TSFunctionPost


@dataclass
class Algorithm:
    name: str
    main: Adapter
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False

    def prepare(self):
        self.main.prepare()

    def finalize(self):
        self.main.finalize()
