import logging
from pathlib import Path
from types import TracebackType
from typing import ContextManager, Union, Optional, Type, NamedTuple, List

import numpy as np
import pandas as pd

from .datasets import Datasets
from .metadata import DatasetId


class DatasetRecord(NamedTuple):
    collection_name: str
    dataset_name: str
    train_path: Optional[str]
    test_path: str
    dataset_type: str
    datetime_index: bool
    split_at: int
    train_type: str
    train_is_normal: bool
    input_type: str
    length: int
    dimensions: int
    contamination: float
    num_anomalies: int
    min_anomaly_length: int
    median_anomaly_length: int
    max_anomaly_length: int
    mean: float
    stddev: float
    trend: str
    stationarity: str
    period_size: Optional[int]


class DatasetManager(ContextManager['DatasetManager'], Datasets):
    """Manages benchmark datasets and their meta-information.

    Manages dataset collections and their meta-information that are stored in a single folder with an index file.
    You can also use this class to create a new TimeEval dataset collection.

    Warnings
    --------
    ATTENTION: Not multi-processing-safe! There is no check for changes to the underlying *dataset.csv* file while
    this class is loaded.

    Read-only access is fine with multiple processes.

    Parameters
    ----------
    data_folder : path
        Path to the folder, where the benchmark data is stored. This folder consists of the file *datasets.csv* and
        the datasets in an hierarchical storage layout.
    custom_datasets_file : path
        Path to a file listing additional custom datasets.
    create_if_missing : bool
        Create an index-file in the `data_folder` if none could be found. Set this to ``False`` if an exception should
        be raised if the folder is wrong or does not exist.

    Raises
    ------
    FileNotFoundError
        If `create_if_missing` is set to ``False`` and no *datasets.csv*-file was found in the `data_folder`.

    See Also
    --------
    :class:`timeeval.datasets.Datasets`
    :class:`timeeval.datasets.MultiDatasetManager`
    """

    def __init__(self, data_folder: Union[str, Path], custom_datasets_file: Optional[Union[str, Path]] = None,
                 create_if_missing: bool = True):
        self._log_: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._filepath = Path(data_folder) / self.INDEX_FILENAME
        self._dirty = False
        if not self._filepath.exists():
            if create_if_missing:
                df = self._create_index_file(self._filepath)
            else:
                raise FileNotFoundError(f"Could not find the index file ({self._filepath.resolve()}). "
                                        "Is your data_folder correct?")
        else:
            df = self._load_df()
        super().__init__(df, custom_datasets_file)

    def __enter__(self) -> 'DatasetManager':
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException],
                 exception_traceback: Optional[TracebackType]) -> Optional[bool]:
        self.save()
        return None

    def _load_df(self) -> pd.DataFrame:
        """Re-read the benchmark dataset collection information from the `datasets.csv` file."""
        return pd.read_csv(self._filepath, index_col=["collection_name", "dataset_name"]).sort_index()

    ### begin overwrites
    @property
    def _log(self) -> logging.Logger:
        return self._log_

    def refresh(self, force: bool = False) -> None:
        if not force and self._dirty:
            raise Exception("There are unsaved changes in memory that would get lost by reading from disk again!")
        else:
            self._df = self._load_df()

    def _get_dataset_path_internal(self, dataset_id: DatasetId, train: bool = False) -> Path:
        path = self._get_value_internal(dataset_id, "train_path" if train else "test_path")
        if not path or (isinstance(path, (np.float64, np.int64, float)) and np.isnan(path)):
            raise KeyError(f"Path to {'training' if train else 'testing'} dataset {dataset_id} not found!")
        return self._filepath.parent.resolve() / path
    ### end overwrites

    def add_dataset(self, dataset: DatasetRecord) -> None:
        """Adds a new dataset to the benchmark dataset collection (in-memory).

        The provided dataset metadata is added to this dataset collection (to the in-memory index). You can save the
        in-memory index to disk using the :func:`timeeval.datasets.DatasetManager.save`-method. The referenced time
        series files (training and testing paths) are not touched. If the same dataset ID
        (collection_name, dataset_name) than an existing dataset is specified, its entries are overwritten!

        Parameters
        ----------
        dataset: DatasetRecord object
            The dataset information to add to the benchmark collection.
        """
        df_new = pd.DataFrame({
            "train_path": dataset.train_path,
            "test_path": dataset.test_path,
            "dataset_type": dataset.dataset_type,
            "datetime_index": dataset.datetime_index,
            "split_at": dataset.split_at,
            "train_type": dataset.train_type,
            "train_is_normal": dataset.train_is_normal,
            "input_type": dataset.input_type,
            "length": dataset.length,
            "dimensions": dataset.dimensions,
            "contamination": dataset.contamination,
            "num_anomalies": dataset.num_anomalies,
            "min_anomaly_length": dataset.min_anomaly_length,
            "median_anomaly_length": dataset.median_anomaly_length,
            "max_anomaly_length": dataset.max_anomaly_length,
            "mean": dataset.mean,
            "stddev": dataset.stddev,
            "trend": dataset.trend,
            "stationarity": dataset.stationarity,
            "period_size": dataset.period_size
        }, index=[(dataset.collection_name, dataset.dataset_name)])
        df = pd.concat([self._df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self._df = df.sort_index()
        self._dirty = True

    def add_datasets(self, datasets: List[DatasetRecord]) -> None:
        """Add a list of datasets to the dataset collection.

        Add a list of new datasets to the benchmark dataset collection (in-memory). Already existing keys are
        overwritten!

        Parameters
        ----------
        datasets: list of DatasetRecord objects
            List of dataset metdata to add to this dataset collection.

        See Also
        --------
        :func:`timeeval.datasets.DatasetManager.add_dataset`
        """
        df_new = pd.DataFrame(datasets)
        df_new.set_index(["collection_name", "dataset_name"], inplace=True)
        df = pd.concat([self._df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self._df = df.sort_index()
        self._dirty = True

    def save(self) -> None:
        """Saves the in-memory dataset index to disk.

        Persists newly added benchmark datasets from memory to the benchmark dataset collection file `datasets.csv`.
        Custom datasets are excluded from persistence and cannot be saved to disk; use
        :func:`timeeval.datasets.DatasetManager.add_dataset` or :func:`timeeval.datasets.DatasetManager.add_datasets`
        to add datasets to the benchmark dataset collection.
        """
        self._df.to_csv(self._filepath)
        self._dirty = False
