from enum import Enum
from pathlib import Path
from typing import Union, Callable

import numpy as np


class TrainingType(Enum):
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi-supervised"
    SUPERVISED = "supervised"

    @staticmethod
    def from_text(name: str) -> 'TrainingType':
        if name.lower() == TrainingType.SUPERVISED.value:
            return TrainingType.SUPERVISED
        elif name.lower() == TrainingType.SEMI_SUPERVISED.value:
            return TrainingType.SEMI_SUPERVISED
        else:  # if name.lower() == TrainingType.UNSUPERVISED.value:
            return TrainingType.UNSUPERVISED


class InputDimensionality(Enum):
    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"

    @staticmethod
    def from_dimensions(n: int) -> 'InputDimensionality':
        if n < 1:
            raise ValueError(f"Zero dimensional dataset is not supported!")
        elif n == 1:
            return InputDimensionality.UNIVARIATE
        else:
            return InputDimensionality.MULTIVARIATE


class ExecutionType(Enum):
    TRAIN = "train"
    EXECUTE = "execute"


AlgorithmParameter = Union[np.ndarray, Path]
TSFunction = Callable[[AlgorithmParameter, dict], AlgorithmParameter]
TSFunctionPost = Union[
    Callable[[AlgorithmParameter, dict], np.ndarray],
    Callable[[np.ndarray, dict], np.ndarray],
    Callable[[Path, dict], np.ndarray]
]
