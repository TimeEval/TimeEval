import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import TracebackType
from typing import Final, ContextManager, Optional, List, Type, Union, NamedTuple

import numpy as np
import pandas as pd

from timeeval.data_types import TrainingType, InputDimensionality
from timeeval.datasets.analyzer import DatasetAnalyzer
from timeeval.datasets.custom import CustomDatasets
from timeeval.datasets.custom_base import CustomDatasetsBase
from timeeval.datasets.custom_noop import NoOpCustomDatasets
from timeeval.datasets.metadata import DatasetId, DatasetMetadata


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


@dataclass
class Dataset:
    datasetId: DatasetId
    dataset_type: str
    training_type: TrainingType
    length: int
    dimensions: int
    num_anomalies: Optional[int] = None

    @property
    def name(self) -> str:
        return self.datasetId[1]

    @property
    def collection_name(self):
        return self.datasetId[0]

    @property
    def input_dimensionality(self) -> InputDimensionality:
        return InputDimensionality.from_dimensions(self.dimensions)

    @property
    def has_anomalies(self) -> Optional[bool]:
        if self.num_anomalies is None:
            return None
        else:
            return self.num_anomalies > 0


class Datasets(ContextManager['Datasets']):
    """
    Manages benchmark datasets and their meta-information.

    ATTENTION: Not multi-processing-safe!
    There is no check for changes to the underlying `dataset.csv` file while this class is loaded.

    Read-only access is fine with multiple processes.
    """

    INDEX_FILENAME: Final[str] = "datasets.csv"
    METADATA_FILENAME_PREFIX: Final[str] = "metadata.json"

    _filepath: Path
    _dirty: bool
    _df: pd.DataFrame
    _custom_datasets: CustomDatasetsBase

    def __init__(self, data_folder: Union[str, Path], custom_datasets_file: Optional[Union[str, Path]] = None):
        """
        :param data_folder: Path to the folder, where the benchmark data is stored.
          This folder consists of the file `datasets.csv` and the datasets in an hierarchical storage layout.
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self._filepath = Path(data_folder) / self.INDEX_FILENAME
        self._dirty = False
        if not self._filepath.exists():
            self._df = self._create_index_file()
        else:
            self.refresh(force=True)

        if custom_datasets_file:
            self.load_custom_datasets(custom_datasets_file)
        else:
            self._custom_datasets = NoOpCustomDatasets()

    def __enter__(self) -> 'Datasets':
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException],
                 exception_traceback: Optional[TracebackType]) -> Optional[bool]:
        self.save()
        return None

    def __repr__(self) -> str:
        return f"{repr(self._df)}\nCustom datasets:\n{repr(self._custom_datasets)}"

    def __str__(self) -> str:
        return f"{str(self._df)}\nCustom datasets:\n{str(self._custom_datasets)}"

    def _create_index_file(self) -> pd.DataFrame:
        df_temp = pd.DataFrame(
            columns=["dataset_name", "collection_name", "train_path", "test_path", "dataset_type", "datetime_index",
                     "split_at", "train_type", "train_is_normal", "input_type", "length", "dimensions", "contamination",
                     "num_anomalies", "min_anomaly_length", "median_anomaly_length", "max_anomaly_length", "mean",
                     "stddev", "trend", "stationarity"])
        df_temp.set_index(["collection_name", "dataset_name"], inplace=True)
        dataset_dir = self._filepath.parent
        if not dataset_dir.is_dir():
            print(f"Directory {dataset_dir} does not exist, creating it!")
            dataset_dir.mkdir(parents=True)
        df_temp.to_csv(self._filepath)
        return df_temp

    def _get_value_internal(self, dataset_id: DatasetId, column_name: str):
        try:
            return self._df.loc[dataset_id, column_name]
        except KeyError as e:
            raise KeyError(f"Dataset {dataset_id} was not found!") from e

    def _build_custom_df(self):
        def safe_extract_path(name: str, train: bool) -> Optional[str]:
            try:
                return str(self._custom_datasets.get_path(name, train))
            except ValueError:
                return None

        collection_names = self._custom_datasets.get_collection_names()
        if len(collection_names) > 0:
            datasets = self._custom_datasets.get_dataset_names()
            indices = [(collection_names[0], name) for name in datasets]
            data = {
                "test_path": [safe_extract_path(name, train=False) for name in datasets],
                "train_path": [safe_extract_path(name, train=True) for name in datasets]
            }
            return pd.DataFrame(data, index=indices, columns=self._df.columns)
        else:
            return pd.DataFrame()

    def add_dataset(self, dataset: DatasetRecord) -> None:
        """
        Add a new dataset to the benchmark dataset collection (in-memory).
        If the same dataset ID (collection_name, dataset_name) is specified than an existing dataset,
        its entries are overwritten!
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
        }, index=[(dataset.collection_name, dataset.dataset_name)])
        df = pd.concat([self._df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self._df = df.sort_index()
        self._dirty = True

    def add_datasets(self, datasets: List[DatasetRecord]) -> None:
        """
        Add a list of new datasets to the benchmark dataset collection (in-memory).
        Already existing keys are overwritten!
        """
        df_new = pd.DataFrame(datasets)
        df_new.set_index(["collection_name", "dataset_name"], inplace=True)
        df = pd.concat([self._df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self._df = df.sort_index()
        self._dirty = True

    def refresh(self, force: bool = False) -> None:
        """Re-read the benchmark dataset collection information from the `datasets.csv` file."""
        if not force and self._dirty:
            raise Exception("There are unsaved changes in memory that would get lost by reading from disk again!")
        else:
            df = pd.read_csv(self._filepath, index_col=["collection_name", "dataset_name"])
            self._df = df.sort_index()

    def save(self) -> None:
        """
        Persist newly added benchmark datasets from memory to the benchmark dataset collection file `datasets.csv`.
        Custom datasets are excluded from persistence and cannot be saved to disk;
        use :py:meth:`Datasets.add_dataset` or :py:meth:`Datasets.add_datasets` to add datasets to the
        permanent benchmark dataset collection.
        """
        self._df.to_csv(self._filepath)
        self._dirty = False

    def get_collection_names(self) -> List[str]:
        return self._custom_datasets.get_collection_names() + list(self._df.index.get_level_values(0).unique())

    def get_dataset_names(self) -> List[str]:
        return self._custom_datasets.get_dataset_names() + list(self._df.index.get_level_values(1).unique())

    def select(self,
               collection_name: Optional[str] = None,
               dataset_name: Optional[str] = None,
               dataset_type: Optional[str] = None,
               datetime_index: Optional[bool] = None,
               train_type: Optional[str] = None,
               train_is_normal: Optional[bool] = None,
               input_type: Optional[str] = None
               ) -> List[DatasetId]:
        """
        Returns a list of dataset identifiers from the benchmark dataset collection whose datasets match **all** of the
        given conditions.

        :param collection_name: restrict datasets to a specific collection
        :param dataset_name: restrict datasets to a specific name
        :param dataset_type: restrict dataset type (e.g. "real" or "synthetic")
        :param datetime_index: only select datasets for which a datetime index exists;
          if `true`: "timestamp"-column has datetime values;
          if `false`: "timestamp"-column has monotonically increasing integer values
        :param train_type: select datasets for specific training needs: "supervised", "semi-supervised", or "unsupervised"
        :param train_is_normal:
          if `true`: only return datasets for which the training dataset does not contain anomalies;
          if `false`: only return datasets for which the training dataset contains anomalies
        :param input_type: restrict dataset to input type: "univariate" or "multivariate"
        :return: list of dataset identifiers (combination of collection name and dataset name)
        """

        def any_selector() -> bool:
            return bool(dataset_type or datetime_index or train_type or train_is_normal or input_type)

        if collection_name in self._custom_datasets.get_collection_names():
            names = self._custom_datasets.get_dataset_names()
            if any_selector() or (dataset_name and dataset_name not in names):
                return []
            elif dataset_name and dataset_name in names:
                return [(collection_name, dataset_name)]
            else:
                return [(collection_name, name) for name in names]
        else:
            # if any selector is applied, there are no matches in custom datasets by definition!
            if any_selector():
                custom_datasets = []
            else:
                custom_datasets = [
                    ("custom", name) for name in self._custom_datasets.get_dataset_names() if name == dataset_name
                ]

            df = self._df  # self.df()
            selectors: List[np.ndarray] = []
            if dataset_type is not None:
                selectors.append(df["dataset_type"] == dataset_type)
            if datetime_index is not None:
                selectors.append(df["datetime_index"] == datetime_index)
            if train_type is not None:
                selectors.append(df["train_type"] == train_type)
            if train_is_normal is not None:
                selectors.append(df["train_is_normal"] == train_is_normal)
            if input_type is not None:
                selectors.append(df["input_type"] == input_type)
            default_mask = np.full(len(df), True)
            mask = reduce(lambda x, y: np.logical_and(x, y), selectors, default_mask)
            bench_datasets = (df[mask]
                              .loc[(slice(collection_name, collection_name), slice(dataset_name, dataset_name)), :]
                              .index
                              .to_list())

            return custom_datasets + bench_datasets

    def df(self) -> pd.DataFrame:
        """Returns a copy of the internal dataset metadata collection."""
        df = pd.concat([self._df, self._build_custom_df()], axis=0, copy=True)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    def load_custom_datasets(self, file_path: Union[str, Path]) -> None:
        self._custom_datasets = CustomDatasets(file_path)

    def get(self, collection_name: Union[str, DatasetId], dataset_name: Optional[str] = None) -> Dataset:
        if isinstance(collection_name, tuple):
            index = collection_name
        elif isinstance(collection_name, str) and dataset_name is not None:
            index = (collection_name, dataset_name)
        else:
            raise ValueError(f"Cannot use {collection_name} and {dataset_name} as index!")

        if index[0] in self._custom_datasets.get_collection_names():
            self.log.warning(f"Custom datasets lack all meta information! "
                             f"Assuming {TrainingType.UNSUPERVISED} and {InputDimensionality.UNIVARIATE} for {index}")
            return Dataset(
                datasetId=index,
                dataset_type="custom",
                training_type=TrainingType.UNSUPERVISED,
                dimensions=1,
                length=-1,
                num_anomalies=-1
            )
            # raise NotImplementedError("Custom datasets lack all meta information!")
        else:
            entry = self._df.loc[index]
            training_type = self.get_training_type(index)
            return Dataset(
                datasetId=index,
                dataset_type=entry["dataset_type"],
                training_type=training_type,
                length=entry["length"],
                dimensions=entry["dimensions"],
                num_anomalies=entry["num_anomalies"]
            )

    def get_dataset_path(self, dataset_id: DatasetId, train: bool = False) -> Path:
        collection_name, dataset_name = dataset_id
        if collection_name in self._custom_datasets.get_collection_names():
            data_path = self._custom_datasets.get_path(dataset_name, train)
            if not data_path:
                raise KeyError(f"Path to {'training' if train else 'testing'} dataset {dataset_id} not found!")
            return data_path.absolute()
        else:
            path = self._get_value_internal(dataset_id, "train_path" if train else "test_path")
            if not path or (isinstance(path, (np.float64, np.int64, float)) and np.isnan(path)):
                raise KeyError(f"Path to {'training' if train else 'testing'} dataset {dataset_id} not found!")
            return self._filepath.parent.absolute() / path

    def get_dataset_df(self, dataset_id: DatasetId, train: bool = False) -> pd.DataFrame:
        path = self.get_dataset_path(dataset_id, train)
        if (dataset_id[0] not in self._custom_datasets.get_collection_names()
                and self._get_value_internal(dataset_id, "datetime_index")):
            return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
        else:
            return pd.read_csv(path)

    def get_dataset_ndarray(self, dataset_id: DatasetId, train: bool = False) -> np.ndarray:
        path = self.get_dataset_path(dataset_id, train)
        # TODO: supply numpy with type information, especially for datasets with datetime_index == True
        return np.genfromtxt(path, delimiter=",", skip_header=1)

    def get_training_type(self, dataset_id: DatasetId) -> TrainingType:
        collection_name, dataset_name = dataset_id
        if collection_name in self._custom_datasets.get_collection_names():
            self.log.warning(f"Custom datasets lack all meta information! "
                             f"Assuming {TrainingType.UNSUPERVISED} for {dataset_id}")
            return TrainingType.UNSUPERVISED
        else:
            train_is_normal = self._get_value_internal(dataset_id, "train_is_normal")
            train_type_name = self._get_value_internal(dataset_id, "train_type")
            training_type = TrainingType.from_text(train_type_name)
            if training_type == TrainingType.SEMI_SUPERVISED and not train_is_normal:
                self.log.warning(f"Dataset {dataset_id} is specified as {training_type} ('train_type'). However, "
                                 f"'train_is_normal' is False! Reverting back to {TrainingType.SUPERVISED}!\n"
                                 f"Please check your dataset configuration!")
                training_type = TrainingType.SUPERVISED
            return training_type

    def get_detailed_metadata(self, dataset_id: DatasetId, train: bool = False) -> DatasetMetadata:
        path = self.get_dataset_path(dataset_id, train)
        metadata_file = path.parent / f"{dataset_id[1]}.{self.METADATA_FILENAME_PREFIX}"
        if metadata_file.exists():
            try:
                return DatasetAnalyzer.load_from_json(metadata_file, train)
            except ValueError:
                self.log.debug(f"Metadata file existed, but the requested file info was not found, recreating it.")
        else:
            self.log.debug(
                f"No metadata file for {dataset_id} exists. Analyzing dataset on-the-fly and storing result.")
        dm = DatasetAnalyzer(dataset_id, is_train=train, dataset_path=path)
        if dataset_id[0] in self._custom_datasets.get_collection_names():
            self.log.warning("Cannot store metadata information for custom datasets!")
        else:
            dm.save_to_json(metadata_file)
        return dm.metadata
