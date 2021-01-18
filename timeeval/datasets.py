from functools import reduce
from pathlib import Path
from types import TracebackType
from typing import Final, ContextManager, Optional, Tuple, List, Type, NamedTuple, Union

import numpy as np
import pandas as pd

from timeeval.customdatasets import CustomDatasets, NoOpCustomDatasets


class DatasetRecord(NamedTuple):
    collection_name: str
    dataset_name: str
    train_path: Optional[str]
    test_path: str
    type: str
    datetime_index: bool
    split_at: int
    train_type: str
    train_is_normal: bool
    input_type: str
    length: int


DatasetId = Tuple[str, str]


class Datasets(ContextManager['Datasets']):
    """
    Manages benchmark datasets and their meta-information.

    ATTENTION: Not multi-processing-safe!
    There is no check for changes to the underlying `dataset.csv` file while this class is loaded.

    Read-only access is fine with multiple processes.
    """

    FILENAME: Final[str] = "datasets.csv"

    _filepath: Path
    _dirty: bool
    _df: pd.DataFrame
    _custom_datasets: CustomDatasets

    def __init__(self, data_folder: Union[str, Path], custom_datasets_file: Optional[Union[str, Path]] = None):
        """
        :param data_folder: Path to the folder, where the benchmark data is stored.
          This folder consists of the file `datasets.csv` and the datasets in an hierarchical storage layout.
        """
        self._filepath = Path(data_folder) / self.FILENAME
        self._dirty = False
        if not self._filepath.exists():
            self._df = self._create_metadata_file()
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

    def _create_metadata_file(self) -> pd.DataFrame:
        df_temp = pd.DataFrame(
            columns=["dataset_name", "collection_name", "train_path", "test_path", "type", "datetime_index", "split_at",
                     "train_type", "train_is_normal", "input_type", "length"])
        df_temp.set_index(["collection_name", "dataset_name"], inplace=True)
        dataset_dir = self._filepath.parent
        if not dataset_dir.is_dir():
            print(f"Directory {dataset_dir} does not exist, creating it!")
            dataset_dir.mkdir()
        df_temp.to_csv(self._filepath)
        return df_temp

    def _get_value_internal(self, dataset_id: DatasetId, column_name: str):
        try:
            return self._df.loc[dataset_id, column_name]
        except KeyError as e:
            raise KeyError(f"Dataset {dataset_id} was not found!") from e

    def add_dataset(self,
                    dataset_id: DatasetId,
                    train_path: Optional[str],
                    test_path: str,
                    dataset_type: str,
                    datetime_index: bool,
                    split_at: int,
                    train_type: str,
                    train_is_normal: bool,
                    input_type: str,
                    dataset_length: int
                    ) -> None:
        """
        Add a new dataset to the benchmark dataset collection (in-memory).
        If the same dataset ID (collection_name, dataset_name) is specified than an existing dataset,
        its entries are overwritten!
        """
        df_new = pd.DataFrame({
            "train_path": train_path,
            "test_path": test_path,
            "type": dataset_type,
            "datetime_index": datetime_index,
            "split_at": split_at,
            "train_type": train_type,
            "train_is_normal": train_is_normal,
            "input_type": input_type,
            "length": dataset_length
        }, index=[dataset_id])
        df = pd.concat([self._df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self._df = df
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
        self._df = df
        self._dirty = True

    def refresh(self, force: bool = False) -> None:
        """Re-read the benchmark dataset collection information from the `datasets.csv` file."""
        if not force and self._dirty:
            raise Exception("There are unsaved changes in memory that would get lost by reading from disk again!")
        else:
            self._df = pd.read_csv(self._filepath, index_col=["collection_name", "dataset_name"])

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
        custom_collection: List[str] = ["custom"] if self._custom_datasets.is_loaded() else []
        return custom_collection + list(self._df.index.get_level_values(0))

    def get_dataset_names(self) -> List[str]:
        return self._custom_datasets.get_dataset_names() + list(self._df.index.get_level_values(1))

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
        Returns a list of dataset identifiers from the benchmark dataset collection whose datasets match all of the
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

        # TODO: search custom datasets
        # if collection_name == "custom":
        # else:

        selectors: List[np.ndarray] = []
        if dataset_type is not None:
            selectors.append(self._df["type"] == dataset_type)
        if datetime_index is not None:
            selectors.append(self._df["datetime_index"] == datetime_index)
        if train_type is not None:
            selectors.append(self._df["train_type"] == train_type)
        if train_is_normal is not None:
            selectors.append(self._df["train_is_normal"] == train_is_normal)
        if input_type is not None:
            selectors.append(self._df["input_type"] == input_type)
        default_mask = np.full(len(self._df), True)
        mask = reduce(lambda x, y: np.logical_and(x, y), selectors, default_mask)

        return self._df[mask].loc[(slice(collection_name), slice(dataset_name)), :].index.to_list()

    def df(self) -> pd.DataFrame:
        """Returns a copy of the internal dataset metadata collection."""

        # TODO: append custom datasets first
        both = self._df.copy()
        return both

    def load_custom_datasets(self, file_path: Union[str, Path]) -> None:
        raise NotImplementedError()

    def get_dataset_path(self, dataset_id: DatasetId, train: bool = False) -> Optional[Path]:
        collection_name, dataset_name = dataset_id
        if collection_name == "custom":
            test_path, train_path = self._custom_datasets.get_path(dataset_name)
            if train:
                return train_path
            else:
                return test_path
        else:
            if train:
                return self._filepath.parent / self._get_value_internal(dataset_id, "train_path")
            else:
                return self._filepath.parent / self._get_value_internal(dataset_id, "test_path")

    def get_dataset_df(self, dataset_id: DatasetId, train: bool = False) -> pd.DataFrame:
        path = self.get_dataset_path(dataset_id, train)
        return pd.read_csv(path)

    def get_dataset_ndarray(self, dataset_id: DatasetId, train: bool = False) -> np.ndarray:
        path = self.get_dataset_path(dataset_id, train)
        return np.genfromtxt(path, delimiter=",", skip_header=1)
