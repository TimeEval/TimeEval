import abc
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd


class CustomDatasetsBase(abc.ABC):

    @abc.abstractmethod
    def get_dataset_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def is_loaded(self):
        ...

    @abc.abstractmethod
    def get_path(self, dataset_name: str) -> Tuple[Path, Optional[Path]]:
        ...

    @abc.abstractmethod
    def load_df(self, dataset_name: str) -> pd.DataFrame:
        ...
