from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from timeeval import Algorithm, AlgorithmParameter, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter


class Baselines:
    @staticmethod
    def random(input_dimensionality: InputDimensionality = InputDimensionality.MULTIVARIATE) -> Algorithm:
        """Random baseline assigns random scores (between 0 and 1)"""
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Random baseline requires an np.ndarray as input!")
            return np.random.default_rng().uniform(0, 1, X.shape[0])

        return Algorithm(
            name="Random",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=input_dimensionality,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def normal(input_dimensionality: InputDimensionality = InputDimensionality.MULTIVARIATE) -> Algorithm:
        """Normal baseline declares everything as normal (score for all points=0)"""
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Normal baseline requires an np.ndarray as input!")
            return np.zeros(X.shape[0])

        return Algorithm(
            name="normal",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=input_dimensionality,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def increasing(input_dimensionality: InputDimensionality = InputDimensionality.MULTIVARIATE) -> Algorithm:
        """Increasing baseline assigns linearly increasing scores between 0 and 1"""
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("Increasing baseline requires an np.ndarray as input!")
            indices = np.arange(X.shape[0])
            return MinMaxScaler().fit_transform(indices.reshape(-1, 1)).reshape(-1)

        return Algorithm(
            name="increasing",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=input_dimensionality,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )
