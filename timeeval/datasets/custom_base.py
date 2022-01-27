import abc
from pathlib import Path
from typing import List, Optional

from .dataset import Dataset
from .metadata import DatasetId
from ..data_types import TrainingType, InputDimensionality


class CustomDatasetsBase(abc.ABC):
    """API definition for custom datasets.

    Internal API! You should **not need to use or modify** this class.
    """

    @abc.abstractmethod
    def get_collection_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_dataset_names(self) -> List[str]:
        ...

    @abc.abstractmethod
    def get_path(self, dataset_name: str, train: bool) -> Path:
        ...

    @abc.abstractmethod
    def get(self, dataset_name: str) -> Dataset:
        ...

    @abc.abstractmethod
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
        ...
