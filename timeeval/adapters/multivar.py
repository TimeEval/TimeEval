from enum import Enum
from typing import Dict, List, Any, Union, Optional
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

    def __call__(self, data: Union[List[np.ndarray], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, list):
            data = pd.DataFrame(np.stack(data, axis=1))
        
        if self == self.MEAN:
            return data.mean(axis=1)
        elif self == self.MEDIAN:
            return data.median(axis=1)
        elif self == self.MAX:
            return data.max(axis=1)
        else:  # self == self.SUM_BEFORE
            return data.sum(axis=1)

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

    def _get_timeseries(self, dataset: AlgorithmParameter) -> pd.DataFrame:
        """Returns the timeseries as a pandas DataFrame."""
        if isinstance(dataset, Path):
            return pd.read_csv(dataset, index_col=0)
        elif isinstance(dataset, np.ndarray):
            return pd.DataFrame(dataset)
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

    def _split_timeseries_into_channels(self, dataset: AlgorithmParameter, tmp_dir: tempfile.TemporaryDirectory) -> List[AlgorithmParameter]:
        """Splits the timeseries into channels and stores the paths to the channels in self._channel_paths."""
        channel_paths: List[Path] = []
        df = self._get_timeseries(dataset)
        for c, column_name in enumerate(df.columns):
            channel_path = Path(tmp_dir) / f"channel_{c}.csv"
            df[[column_name]].to_csv(channel_path)
            channel_paths.append(channel_path)
        return channel_paths
    
    def _combine_channels(self, dataset: AlgorithmParameter, tmp_dir: tempfile.TemporaryDirectory) -> AlgorithmParameter:
        """Combines the channels into a single timeseries and stores the path to the timeseries in self._channel_paths."""
        channel_path = Path(tmp_dir) / "combined_test.csv"
        combined = self._aggregation(self._get_timeseries(dataset))
        combined.to_csv(channel_path)
        return channel_path

    def _combine_channel_scores(self, scores: List[AlgorithmParameter]) -> AlgorithmParameter:
        """Combines the scores of the channels into a single score file."""
        scores = pd.concat([self._get_timeseries(score) for score in scores], axis=1)
        scores = self._aggregation(scores)
        return scores.values[:, 0]

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        """Calls the anomaly detector on each channel and aggregates the results or combines the channels and calls the anomaly detector on the combined timeseries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_paths: Optional[List[Path]] = None
            if self._aggregation.combining_before:
                dataset_paths = [self._combine_channels(dataset, tmp_dir)]
            else:
                dataset_paths = self._split_timeseries_into_channels(dataset, tmp_dir)
            
            scores: List[AlgorithmParameter] = []
            for dataset_path in dataset_paths:
                args["dataset"] = dataset_path
                scores.append(self._docker_adapter._call(dataset, args))
            
            if not self._aggregation.combining_before:
                return self._combine_channel_scores(scores)
            assert len(scores) == 1, "Expected only one score file when combining before"
            return scores[0]
