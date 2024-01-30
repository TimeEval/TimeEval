from enum import Enum
from typing import Callable, Dict, List, Any, Union, Optional
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .base import Adapter
from ..data_types import AlgorithmParameter


class AggregationMethod(Enum):
    """An enum that specifies how to aggregate the anomaly scores of the channels."""
    MEAN = 0
    """aggregates channel scores using the element-wise mean."""
    MEDIAN = 1
    """aggregates channel scores using the element-wise median."""
    MAX = 2
    """aggregates channel scores using the element-wise max."""
    SUM_BEFORE = 3
    """sums the channels before running the anomaly detector."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Aggregates the channels using the specified method.

        :meta private:
        """

        if self == self.MEAN:
            return data.mean(axis=1)  # type: ignore
        elif self == self.MEDIAN:
            return np.median(data, axis=1)  # type: ignore
        elif self == self.MAX:
            return data.max(axis=1)  # type: ignore
        else:  # self == self.SUM_BEFORE
            return data.sum(axis=1)  # type: ignore

    @property
    def combining_before(self) -> bool:
        """
        Returns whether the aggregation method combines the channels before or after running the anomaly detector.

        :meta private:
        """
        return self == self.SUM_BEFORE


class MultivarAdapter(Adapter):
    """
    An adapter that allows to apply univariate anomaly detectors to multiple dimensions of a timeseries.
    In one case, the adapter runs the anomaly detector on each dimension separately and aggregates the results using the specified aggregation method.
    In the other case, the adapter combines the dimensions into a single timeseries and runs the anomaly detector on the combined timeseries.

    Parameters
    ----------

    adapter : Adapter
        The :class:`~timeeval.adapters.Adapter` that runs the anomaly detector on each dimension.

    aggregation : AggregationMethod
        The :class:`~timeeval.adapters.multivar.AggregationMethod` to use to combine the anomaly scores of the dimensions.
    """

    def __init__(self, adapter: Adapter, aggregation: AggregationMethod = AggregationMethod.MEAN) -> None:
        assert not isinstance(adapter, MultivarAdapter), "Cannot nest MultivarAdapters"

        self._adapter = adapter
        self._aggregation = aggregation

    def _get_timeseries(self, dataset: AlgorithmParameter) -> pd.DataFrame:
        """Returns the timeseries as a pandas DataFrame."""

        df: Optional[pd.DataFrame] = None
        if isinstance(dataset, Path):
            df = pd.read_csv(dataset, index_col=0)
        elif isinstance(dataset, np.ndarray):
            df = pd.DataFrame(dataset)
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        return df

    def _split_timeseries_into_channels(self, dataset: AlgorithmParameter, tmp_dir: Path) -> List[Path]:
        """Splits the timeseries into channels and stores the paths to the channels in self._channel_paths."""
        channel_paths: List[Path] = []
        df = self._get_timeseries(dataset)
        anomaly_column = df.columns[-1]
        for c, column_name in enumerate(df.columns[:-1]):
            channel_path = tmp_dir / f"channel_{c}.csv"
            df[[column_name, anomaly_column]].to_csv(channel_path)
            channel_paths.append(channel_path)
        return channel_paths

    def _combine_channels(self, dataset: AlgorithmParameter, tmp_dir: Path) -> Path:
        """Combines the channels into a single timeseries and stores the path to the timeseries in self._channel_paths."""
        channel_path = tmp_dir / "combined_test.csv"
        loaded = self._get_timeseries(dataset)
        combined = pd.DataFrame(self._aggregation(loaded.iloc[:, :-1].values), index=loaded.index, columns=["combined"])
        combined[loaded.columns[-1]] = loaded.iloc[:, -1]
        combined.to_csv(channel_path)
        return channel_path

    def _combine_channel_scores(self, scores: List[AlgorithmParameter]) -> np.ndarray:
        """Combines the scores of the channels into a single score file."""
        loaded_scores = np.concatenate([self._get_timeseries(score).values for score in scores], axis=1)
        combined_scores: np.ndarray = self._aggregation(loaded_scores)
        return combined_scores

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        """Calls the anomaly detector on each channel and aggregates the results or combines the channels and calls the anomaly detector on the combined timeseries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_paths: Optional[List[Path]] = None
            if self._aggregation.combining_before:
                dataset_paths = [self._combine_channels(dataset, Path(tmp_dir))]
            else:
                dataset_paths = self._split_timeseries_into_channels(dataset, Path(tmp_dir))

            scores: List[AlgorithmParameter] = []
            for dataset_path in dataset_paths:
                scores.append(self._adapter._call(dataset_path, args))

            if not self._aggregation.combining_before:
                return self._combine_channel_scores(scores)
            assert len(scores) == 1, "Expected only one score file when combining before"
            return scores[0]

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        return self._adapter.get_prepare_fn()

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        return self._adapter.get_finalize_fn()
