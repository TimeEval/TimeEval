from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from timeeval import Algorithm, AlgorithmParameter, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter


class Baselines:
    @staticmethod
    def random() -> Algorithm:
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Random baseline requires an np.ndarray as input!")
            return np.random.default_rng().uniform(0, 1, X.shape[0])

        return Algorithm(
            name="Random",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def normal() -> Algorithm:
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Normal baseline requires an np.ndarray as input!")
            return np.zeros(X.shape[0])

        return Algorithm(
            name="normal",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def increasing() -> Algorithm:
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Increasing baseline requires an np.ndarray as input!")
            indices = np.arange(X.shape[0])
            return MinMaxScaler().fit_transform(indices.reshape(-1, 1)).reshape(-1)

        return Algorithm(
            name="increasing",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )
