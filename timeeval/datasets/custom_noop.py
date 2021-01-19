from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from timeeval.datasets.custom_base import CustomDatasetsBase


class NoOpCustomDatasets(CustomDatasetsBase):

    def __init__(self):
        super().__init__()

    def is_loaded(self):
        return False

    def get_path(self, dataset_config: Path) -> Tuple[Path, Optional[Path]]:
        raise KeyError("No custom datasets loaded!")

    def load_df(self, dataset_name: str) -> pd.DataFrame:
        raise KeyError("No custom datasets loaded!")

    def get_dataset_names(self) -> List[str]:
        return []
