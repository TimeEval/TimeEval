import abc
import logging
from functools import reduce
from pathlib import Path
from typing import Final, List, Union, Optional

import numpy as np
import pandas as pd

from ..data_types import TrainingType, InputDimensionality
from .analyzer import DatasetAnalyzer
from .custom import CustomDatasets
from .custom_base import CustomDatasetsBase
from .custom_noop import NoOpCustomDatasets
from .dataset import Dataset
from .metadata import DatasetId, DatasetMetadata


class Datasets(abc.ABC):
    """
    Provides read-only access to benchmark datasets and their meta-information.
    """

    INDEX_FILENAME: Final[str] = "datasets.csv"
    METADATA_FILENAME_SUFFIX: Final[str] = "metadata.json"

    def __init__(self, df: pd.DataFrame, custom_datasets_file: Optional[Union[str, Path]] = None):
        """
        :param custom_datasets_file: Path to a file listing additional custom datasets.
        """
        self._df: pd.DataFrame = df

        if custom_datasets_file:
            self.load_custom_datasets(custom_datasets_file)
        else:
            self._custom_datasets: CustomDatasetsBase = NoOpCustomDatasets()

    def __repr__(self) -> str:
        return f"{repr(self._df)}\nCustom datasets:\n{repr(self._custom_datasets)}"

    def __str__(self) -> str:
        return f"{str(self._df)}\nCustom datasets:\n{str(self._custom_datasets)}"

    @property
    @abc.abstractmethod
    def _log(self) -> logging.Logger:
        raise NotImplementedError()

    @abc.abstractmethod
    def refresh(self, force: bool = False) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_dataset_path_internal(self, dataset_id: DatasetId, train: bool = False) -> Path:
        raise NotImplementedError()

    def _create_index_file(self, filepath: Path) -> pd.DataFrame:
        df_temp = pd.DataFrame(
            columns=["dataset_name", "collection_name", "train_path", "test_path", "dataset_type", "datetime_index",
                     "split_at", "train_type", "train_is_normal", "input_type", "length", "dimensions", "contamination",
                     "num_anomalies", "min_anomaly_length", "median_anomaly_length", "max_anomaly_length", "mean",
                     "stddev", "trend", "stationarity", "period_size"])
        df_temp.set_index(["collection_name", "dataset_name"], inplace=True)
        dataset_dir = filepath.parent
        if not dataset_dir.is_dir():
            self._log.warning(f"Directory {dataset_dir} does not exist, creating it!")
            dataset_dir.mkdir(parents=True)
        df_temp.to_csv(filepath)
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

    def get_collection_names(self) -> List[str]:
        return self._custom_datasets.get_collection_names() + list(self._df.index.get_level_values(0).unique())

    def get_dataset_names(self) -> List[str]:
        return self._custom_datasets.get_dataset_names() + list(self._df.index.get_level_values(1).unique())

    def select(self,
               collection: Optional[str] = None,
               dataset: Optional[str] = None,
               dataset_type: Optional[str] = None,
               datetime_index: Optional[bool] = None,
               training_type: Optional[TrainingType] = None,
               train_is_normal: Optional[bool] = None,
               input_dimensionality: Optional[InputDimensionality] = None,
               min_anomalies: Optional[int] = None,
               max_anomalies: Optional[int] = None,
               max_contamination: Optional[float] = None
               ) -> List[DatasetId]:
        """
        Returns a list of dataset identifiers from the benchmark dataset collection whose datasets match **all** of the
        given conditions.

        :param collection: restrict datasets to a specific collection
        :param dataset: restrict datasets to a specific name
        :param dataset_type: restrict dataset type (e.g. "real" or "synthetic")
        :param datetime_index: only select datasets for which a datetime index exists;
          if `true`: "timestamp"-column has datetime values;
          if `false`: "timestamp"-column has monotonically increasing integer values
        :param training_type: select datasets for specific training needs:
          "supervised" (`TrainingType.SUPERVISED`),
          "semi-supervised" (`TrainingType.SEMI_SUPERVISED`), or
          "unsupervised" (`TrainingType.UNSUPERVISED`)
        :param train_is_normal:
          if `true`: only return datasets for which the training dataset does not contain anomalies;
          if `false`: only return datasets for which the training dataset contains anomalies
        :param input_dimensionality: restrict dataset to input type: `InputDimensionality.UNIVARIATE` or
          `InputDimensionality.MULTIVARIATE`
        :param min_anomalies: restrict datasets to those with a minimum number of `min_anomalies` anomalous subsequences
        :param max_anomalies: restrict datasets to those with a maximum number of `max_anomalies` anomalous subsequences
        :param max_contamination: restrict datasets to those having a contamination smaller or equal to
          `max_contamination`
        :return: list of dataset identifiers (combination of collection name and dataset name)
        """

        def any_selector() -> bool:
            return bool(dataset_type or datetime_index or training_type or train_is_normal or input_dimensionality)

        if collection in self._custom_datasets.get_collection_names():
            names = self._custom_datasets.get_dataset_names()
            if any_selector() or (dataset and dataset not in names):
                return []
            elif dataset and dataset in names:
                return [(collection, dataset)]
            else:
                return [(collection, name) for name in names]
        else:
            # if any selector is applied, there are no matches in custom datasets by definition!
            # FIXME: this does not apply anymore!
            if any_selector():
                custom_datasets = []
            else:
                custom_datasets = [
                    ("custom", name) for name in self._custom_datasets.get_dataset_names() if name == dataset
                ]

            df = self._df  # self.df()
            selectors: List[np.ndarray] = []
            if dataset_type is not None:
                selectors.append(df["dataset_type"] == dataset_type)
            if datetime_index is not None:
                selectors.append(df["datetime_index"] == datetime_index)
            if training_type is not None:
                # translate to internal representation (strings)
                train_type: str = training_type.value
                selectors.append(df["train_type"] == train_type)
            if train_is_normal is not None:
                selectors.append(df["train_is_normal"] == train_is_normal)
            if input_dimensionality is not None:
                # translate to internal representation (strings)
                input_type: str = input_dimensionality.value
                selectors.append(df["input_type"] == input_type)
            if min_anomalies is not None:
                selectors.append(df["num_anomalies"] >= min_anomalies)
            if max_anomalies is not None:
                selectors.append(df["num_anomalies"] <= max_anomalies)
            if max_contamination is not None:
                selectors.append(df["contamination"] <= max_contamination)
            default_mask = np.full(len(df), True)
            mask = reduce(lambda x, y: np.logical_and(x, y), selectors, default_mask)
            bench_datasets = (df[mask]
                              .loc[(slice(collection, collection), slice(dataset, dataset)), :]
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
            return self._custom_datasets.get(index[1])
        else:
            entry = self._df.loc[index]
            training_type = self.get_training_type(index)
            period = None
            try:
                period = entry["period_size"]
            except KeyError:
                pass
            return Dataset(
                datasetId=index,
                dataset_type=entry["dataset_type"],
                training_type=training_type,
                length=entry["length"],
                dimensions=entry["dimensions"],
                contamination=entry["contamination"],
                num_anomalies=entry["num_anomalies"],
                min_anomaly_length=entry["min_anomaly_length"],
                median_anomaly_length=entry["median_anomaly_length"],
                max_anomaly_length=entry["max_anomaly_length"],
                period_size=period
            )

    def get_dataset_path(self, dataset_id: DatasetId, train: bool = False) -> Path:
        collection_name, dataset_name = dataset_id
        if collection_name in self._custom_datasets.get_collection_names():
            return self._custom_datasets.get_path(dataset_name, train)
        else:
            return self._get_dataset_path_internal(dataset_id, train)

    def get_dataset_df(self, dataset_id: DatasetId, train: bool = False) -> pd.DataFrame:
        path = self.get_dataset_path(dataset_id, train)
        if dataset_id[0] not in self._custom_datasets.get_collection_names():
            if self._get_value_internal(dataset_id, "datetime_index"):
                return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
            else:
                return pd.read_csv(path)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
            # timestamp parsing failed, hopefully because we have an integer-timestamp
            if df["timestamp"].dtype == np.dtype("O"):
                try:
                    df["timestamp"] = df["timestamp"].astype(np.int_)
                except ValueError as e:
                    raise TypeError(
                        f"Incorrect timestamp format (expected valid date or integer index) of dataset {dataset_id}"
                    ) from e
            return df

    def get_dataset_ndarray(self, dataset_id: DatasetId, train: bool = False) -> np.ndarray:
        return self.get_dataset_df(dataset_id, train).values

    def get_training_type(self, dataset_id: DatasetId) -> TrainingType:
        collection_name, dataset_name = dataset_id
        if collection_name in self._custom_datasets.get_collection_names():
            return self._custom_datasets.get(dataset_name).training_type
        else:
            train_is_normal = self._get_value_internal(dataset_id, "train_is_normal")
            train_type_name = self._get_value_internal(dataset_id, "train_type")
            training_type = TrainingType.from_text(train_type_name)
            if training_type == TrainingType.SEMI_SUPERVISED and not train_is_normal:
                self._log.warning(f"Dataset {dataset_id} is specified as {training_type} ('train_type'). However, "
                                  f"'train_is_normal' is False! Reverting back to {TrainingType.SUPERVISED}!\n"
                                  f"Please check your dataset configuration!")
                training_type = TrainingType.SUPERVISED
            return training_type

    def get_detailed_metadata(self, dataset_id: DatasetId, train: bool = False) -> DatasetMetadata:
        path = self.get_dataset_path(dataset_id, train)
        metadata_file = path.parent / f"{dataset_id[1]}.{self.METADATA_FILENAME_SUFFIX}"
        if metadata_file.exists():
            try:
                return DatasetAnalyzer.load_from_json(metadata_file, train)
            except ValueError:
                self._log.debug(f"Metadata file existed, but the requested file info was not found, recreating it.")
        else:
            self._log.debug(
                f"No metadata file for {dataset_id} exists. Analyzing dataset on-the-fly and storing result.")
        dm = DatasetAnalyzer(dataset_id, is_train=train, dataset_path=path)
        if dataset_id[0] in self._custom_datasets.get_collection_names():
            self._log.warning("Cannot store metadata information for custom datasets!")
        else:
            dm.save_to_json(metadata_file)
        return dm.metadata
