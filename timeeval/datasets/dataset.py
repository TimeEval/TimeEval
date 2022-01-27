from dataclasses import dataclass
from typing import Optional

from ..data_types import TrainingType, InputDimensionality
from .metadata import DatasetId


@dataclass
class Dataset:
    datasetId: DatasetId
    dataset_type: str
    training_type: TrainingType
    length: int
    dimensions: int
    contamination: float
    min_anomaly_length: int
    median_anomaly_length: int
    max_anomaly_length: int
    period_size: Optional[int] = None
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
