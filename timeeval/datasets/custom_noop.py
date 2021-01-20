from pathlib import Path
from typing import List

from timeeval.datasets.custom_base import CustomDatasetsBase


class NoOpCustomDatasets(CustomDatasetsBase):

    def __init__(self):
        super().__init__()

    def get_path(self, dataset_name: str, train: bool) -> Path:
        raise KeyError("No custom datasets loaded!")

    def get_dataset_names(self) -> List[str]:
        return []

    def get_collection_names(self) -> List[str]:
        return []
