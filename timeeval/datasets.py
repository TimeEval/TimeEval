import os
from functools import reduce
from types import TracebackType
from typing import Final, ContextManager, Optional, Tuple, List, Type, NamedTuple, Iterable

import numpy as np
import pandas as pd


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


class Datasets(ContextManager['Datasets']):
    """
    ATTENTION: Not multi-processing-safe!
    There is no check for changes to the underlying `dataset.csv` file while this class is loaded.

    Read-only access is fine with multiple processes.
    """

    FILENAME: Final[str] = "datasets.csv"

    _filepath: str
    _dirty: bool
    df: pd.DataFrame

    def __init__(self, data_folder: str):
        self._filepath = os.path.join(data_folder, self.FILENAME)
        self._dirty = False
        if not os.path.isfile(self._filepath):
            self.df = self._create_metadata_file()
        else:
            self.refresh(force=True)

    def __enter__(self) -> 'Datasets':
        return self

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException],
                 exception_traceback: Optional[TracebackType]) -> Optional[bool]:
        self.save()
        return None

    def __repr__(self) -> str:
        return repr(self.df)

    def __str__(self) -> str:
        return str(self.df)

    def _create_metadata_file(self) -> pd.DataFrame:
        df_temp = pd.DataFrame(
            columns=["dataset_name", "collection_name", "train_path", "test_path", "type", "datetime_index", "split_at",
                     "train_type", "train_is_normal", "input_type", "length"])
        df_temp.set_index(["collection_name", "dataset_name"], inplace=True)
        dataset_dir = os.path.dirname(self._filepath)
        if not os.path.isdir(dataset_dir):
            print(f"Directory {dataset_dir} does not exist, creating it!")
            os.mkdir(dataset_dir)
        df_temp.to_csv(self._filepath)
        return df_temp

    def add_dataset(self,
                    collection_name: str,
                    dataset_name: str,
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
        }, index=[(collection_name, dataset_name)])
        df = pd.concat([self.df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self.df = df
        self._dirty = True

    def add_datasets(self, datasets: List[DatasetRecord]) -> None:
        df_new = pd.DataFrame(datasets)
        df_new.set_index(["collection_name", "dataset_name"], inplace=True)
        df = pd.concat([self.df, df_new], axis=0)
        df = df[~df.index.duplicated(keep="last")]
        self.df = df
        self._dirty = True

    def refresh(self, force: bool = False) -> None:
        if not force and self._dirty:
            raise Exception("There are unsaved changes in memory that would get lost by reading from disk again!")
        else:
            self.df = pd.read_csv(self._filepath, index_col=["collection_name", "dataset_name"])

    def save(self) -> None:
        self.df.to_csv(self._filepath)
        self._dirty = False

    def get_collection_names(self) -> Iterable[str]:
        return self.df.index.get_level_values(0)

    def get_dataset_names(self) -> Iterable[str]:
        return self.df.index.get_level_values(1)

    def select(self,
               collection_name: Optional[str] = None,
               dataset_name: Optional[str] = None,
               dataset_type: Optional[str] = None,
               datetime_index: Optional[bool] = None,
               train_type: Optional[str] = None,
               train_is_normal: Optional[bool] = None,
               input_type: Optional[str] = None
               ) -> List[Tuple[str, str]]:
        selectors: List[np.ndarray] = []
        if dataset_type is not None:
            selectors.append(self.df["type"] == dataset_type)
        if datetime_index is not None:
            selectors.append(self.df["datetime_index"] == datetime_index)
        if train_type is not None:
            selectors.append(self.df["train_type"] == train_type)
        if train_is_normal is not None:
            selectors.append(self.df["train_is_normal"] == train_is_normal)
        if input_type is not None:
            selectors.append(self.df["input_type"] == input_type)
        default_mask = np.full(len(self.df), True)
        mask = reduce(lambda x, y: np.logical_and(x, y), selectors, default_mask)

        return self.df[mask].loc[(slice(collection_name), slice(dataset_name)), :].index.to_list()
