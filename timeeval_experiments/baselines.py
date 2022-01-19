from pathlib import Path
from typing import Dict, Any, Callable

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

    @staticmethod
    def deviating_from_mean() -> Algorithm:
        """
        Baseline that outputs the difference between current value and time series mean as anomaly score.
        If the dataset is multivariate, the mean deviation is used.
        """
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("DeviatingFromMean baseline requires an np.ndarray as input!")
            data = np.asarray(X, dtype=np.float_)
            return Baselines._deviating_from(data, np.nanmean)

        return Algorithm(
            name="DeviatingFromMean",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def deviating_from_median() -> Algorithm:
        """
        Baseline that outputs the difference between current value and time series median as anomaly score.
        If the dataset is multivariate, the mean deviation is used.
        """
        def fn(X: AlgorithmParameter, params: Dict[str, Any]) -> AlgorithmParameter:
            if isinstance(X, Path):
                raise ValueError("DeviatingFromMedian baseline requires an np.ndarray as input!")
            data = np.asarray(X, dtype=np.float_)
            return Baselines._deviating_from(data, np.nanmedian)

        return Algorithm(
            name="DeviatingFromMedian",
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            data_as_file=False,
            main=FunctionAdapter(fn)
        )

    @staticmethod
    def _deviating_from(data: np.ndarray, fn: Callable) -> np.ndarray:
        # univariate
        if len(data.shape) == 1:
            diffs = np.abs(data - fn(data))
            return diffs / diffs.max()
        # multivariate
        else:
            diffs = np.abs(data - fn(data, axis=0))
            diffs = diffs / diffs.max(axis=0)
            return diffs.mean(axis=1)
