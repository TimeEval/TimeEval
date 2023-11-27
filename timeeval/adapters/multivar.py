from enum import Enum
from typing import Callable, Dict, List, Any, Union, Optional
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from timeeval.adapters.docker import DockerAdapter

from .base import Adapter
from ..data_types import AlgorithmParameter


class AggregationMethod(Enum):
    MEAN = 0  # aggregate channel scores using the element-wise mean
    MEDIAN = 1  # aggregate channel scores using the element-wise median
    MAX = 2  # aggregate channel scores using the element-wise max
    SUM_BEFORE = 3  # sum the channels before running the anomaly detector

    def __call__(self, data: Union[List[np.ndarray], pd.DataFrame]) -> np.ndarray:
        if self == self.MEAN:
            fn: Any = np.mean
        elif self == self.MEDIAN:
            fn = np.median
        elif self == self.MAX:
            fn = np.max
        else:  # self == self.SUM_BEFORE
            fn = np.sum

        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.stack(data, axis=1)

        values: np.ndarray = fn(data, axis=1).reshape(-1)
        return values

    @property
    def combining_before(self) -> bool:
        return self == self.SUM_BEFORE
    

class MultivarAdapter(Adapter):
    """An adapter that allows to apply univariate anomaly detectors to multiple dimensions of a timeseries. 
    The adapter runs the anomaly detector on each dimension separately and aggregates the results using the specified aggregation method."""

    def __init__(self, docker_adapter: DockerAdapter, aggregation: AggregationMethod = AggregationMethod.MEAN) -> None:
        """Initializes the adapter. Uses the specified DockerAdapter to run the anomaly detector on each dimension separately."""
        self._docker_adapter = docker_adapter
        self._aggregation = aggregation
        self._tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._channel_paths: List[Path] = []

    def _get_timeseries_path(self) -> Path:
        ...

    def _split_timeseries_into_channels(self) -> None:
        """Splits the timeseries into channels and stores the paths to the channels in self._channel_paths."""
        timeseries_path = self._get_timeseries_path()
        df = pd.read_csv(timeseries_path, index_col=0)
        self._tmp_dir = tempfile.TemporaryDirectory()
        for c, column_name in enumerate(df.columns):
            channel_path = Path(self._tmp_dir) / f"channel_{c}.csv"
            df[[column_name]].to_csv(channel_path)
            self._channel_paths.append(channel_path)
    
    def _combine_channels(self) -> None:
        """Combines the channels into a single timeseries and stores the path to the timeseries in self._channel_paths."""
        timeseries_path = self._get_timeseries_path()
        self._tmp_dir = tempfile.TemporaryDirectory()
        channel_path = Path(self._tmp_dir) / "combined_test.csv"
        combined = self._aggregation(pd.read_csv(timeseries_path, index_col=0))
        df = pd.DataFrame(combined, columns=["combined"])
        df.to_csv(channel_path)

    def _combine_channel_scores(self) -> None:
        """Combines the scores of the channels into a single score file."""
        scores = []
        for channel_path in self._channel_paths:
            scores.append(pd.read_csv(channel_path, index_col=0).values)
        scores = pd.DataFrame(self._aggregation(scores), columns=["score"])
        scores.to_csv(self._docker_adapter._result_path())

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        return self._docker_adapter._call(dataset, args)

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        """Returns a function that is called before the anomaly detector is run. This function is called only once per anomaly detection run."""
        def prepare_fn():
            self._docker_adapter.get_prepare_fn()()
            if self._aggregation.combining_before:
                self._combine_channels()
            else:
                self._split_timeseries_into_channels()
        return prepare_fn
    
    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        """Returns a function that is called after the anomaly detector is run. This function is called only once per anomaly detection run."""
        def finalize_fn():
            self._docker_adapter.get_finalize_fn()()
            if not self._aggregation.combining_before:
                self._combine_channel_scores()
            self._tmp_dir.cleanup()
        return finalize_fn
