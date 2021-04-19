from enum import Enum
from pathlib import Path
from typing import Union, Callable

import numpy as np


class ExecutionType(Enum):
    TRAIN = "train"
    EXECUTE = "execute"


AlgorithmParameter = Union[np.ndarray, Path]
TSFunction = Callable[[AlgorithmParameter, dict], AlgorithmParameter]
TSFunctionPost = Callable[[AlgorithmParameter, dict], np.ndarray]
