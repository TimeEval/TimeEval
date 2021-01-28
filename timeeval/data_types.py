import numpy as np
from pathlib import Path
from typing import Union, Callable


AlgorithmParameter = Union[np.ndarray, Path]
TSFunction = Callable[[AlgorithmParameter, dict], AlgorithmParameter]
TSFunctionPost = Callable[[AlgorithmParameter, dict], np.ndarray]
