import logging
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from .datasets import Datasets
from .metadata import DatasetId


class MultiDatasetManager(Datasets):
    """Provides read-only access to multiple benchmark datasets collections and their meta-information.

    Manages dataset collections and their meta-information that are stored in multiple folders. The entries in all
    index files must be unique and are NOT allowed to overlap! This would lead to information loss!

    Parameters
    ----------
    data_folders : list of paths
        List of data paths that hold the datasets and the index files.
    custom_datasets_file : path
        Path to a file listing additional custom datasets.

    Raises
    ------
    FileNotFoundError
        If the *datasets.csv*-file was not found in any of the `data_folders`.

    See Also
    --------
    :class:`timeeval.datasets.Datasets`
    :class:`timeeval.datasets.DatasetManager`
    """

    def __init__(self, data_folders: List[Union[str, Path]], custom_datasets_file: Optional[Union[str, Path]] = None):
        self._log_: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._filepaths = [Path(folder) / self.INDEX_FILENAME for folder in data_folders]
        existing_files = np.array([p.exists() for p in self._filepaths])
        if not np.all(existing_files):
            missing = np.array(self._filepaths)[~existing_files]
            missing = [str(p) for p in missing]
            raise FileNotFoundError(f"Could not find the index files ({', '.join(missing)}). "
                                    "Is your data_folders parameter correct?")
        else:
            path_mapping, df = self._load_df()
        self._root_path_mapping: Dict[Tuple[str, str], Path] = path_mapping
        super().__init__(df, custom_datasets_file)

    @property
    def _log(self) -> logging.Logger:
        return self._log_

    def _load_df(self) -> Tuple[Dict[Tuple[str, str], Path], pd.DataFrame]:
        """Read the dataset metadata from the index files."""
        df = pd.DataFrame()
        root_path_mapping = {}
        for path in self._filepaths:
            df_new = pd.read_csv(path, index_col=["collection_name", "dataset_name"])
            for item in df_new.index:
                root_path_mapping[item] = path.parent
            df = pd.concat([df, df_new])
        return root_path_mapping, df.sort_index()

    def refresh(self, force: bool = False) -> None:
        # ignore the force parameter
        self._df = self._load_df()

    def _get_dataset_path_internal(self, dataset_id: DatasetId, train: bool = False) -> Path:
        root_path = self._root_path_mapping[dataset_id]
        path = self._get_value_internal(dataset_id, "train_path" if train else "test_path")
        if not path or (isinstance(path, (np.float64, np.int64, float)) and np.isnan(path)):
            raise KeyError(f"Path to {'training' if train else 'testing'} dataset {dataset_id} not found!")
        return root_path.resolve() / path
