from pathlib import Path
from typing import List, Optional

from .custom_base import CustomDatasetsBase
from .dataset import Dataset
from .metadata import DatasetId
from ..data_types import TrainingType, InputDimensionality


class NoOpCustomDatasets(CustomDatasetsBase):
    """Dummy implementation of the CustomDatasets interface.

    Internal API! You should **not need to use or modify** this class.

    This dummy implementation does nothing and improves readability of the
    :class:`timeeval.datasets.Datasets`-implementation by removing the need for None-checks.
    """

    def __init__(self):
        super().__init__()

    def get_path(self, dataset_name: str, train: bool) -> Path:
        raise KeyError("No custom datasets loaded!")

    def get_dataset_names(self) -> List[str]:
        return []

    def get_collection_names(self) -> List[str]:
        return []

    def get(self, dataset_name: str) -> Dataset:
        raise KeyError("No custom datasets loaded!")

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
        return []
