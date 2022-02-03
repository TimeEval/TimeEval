import json
import warnings
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict, NamedTuple

import numpy as np
import pandas as pd

from .analyzer import DatasetAnalyzer
from .custom_base import CustomDatasetsBase
from .dataset import Dataset
from .metadata import DatasetId
from ..data_types import TrainingType, InputDimensionality


TRAIN_PATH_KEY = "train_path"
TEST_PATH_KEY = "test_path"
TYPE_KEY = "type"
PERIOD_KEY = "period"


class CDEntry(NamedTuple):
    test_path: Path
    train_path: Optional[Path]
    details: Dataset


def _dataset_id(name: str, collection_name: str = "custom") -> Tuple[str, str]:
    return collection_name, name


def _training_type(train_path: Optional[Path]) -> TrainingType:
    if train_path is None:
        return TrainingType.UNSUPERVISED
    else:
        labels = pd.read_csv(train_path).iloc[:, -1]
        if np.any(labels):
            return TrainingType.SUPERVISED
        else:
            return TrainingType.SEMI_SUPERVISED


class CustomDatasets(CustomDatasetsBase):
    """Implementation of the custom datasets API.

    Internal API! You should **not need to use or modify** this class.

    This class behaves similar to the :class:`timeeval.datasets.Datasets`-API while using a different internal
    representation for the dataset index.
    """

    def __init__(self, dataset_config: Union[str, Path]):
        super().__init__()
        dataset_config_path = Path(dataset_config)
        with dataset_config_path.open("r") as f:
            config = json.load(f)
        self.root_path: Path = dataset_config_path.parent

        store = {}
        for dataset in config:
            self._validate_dataset(dataset, config[dataset])
            store[dataset] = self._analyze_dataset(dataset, config[dataset])

        self._dataset_store: Dict[str, CDEntry] = store

    def _extract_path(self, obj: dict, key: str) -> Path:
        path_string = obj[key]
        path = self.root_path / path_string
        return path.resolve()

    def _validate_dataset(self, name: str, ds_obj: dict) -> None:
        if TEST_PATH_KEY not in ds_obj:
            raise ValueError(f"The dataset {name} misses the required '{TEST_PATH_KEY}' property.")
        elif not self._extract_path(ds_obj, TEST_PATH_KEY).exists():
            raise ValueError(f"The test file for dataset {name} was not found (property '{TEST_PATH_KEY}')!")
        if TRAIN_PATH_KEY in ds_obj and not self._extract_path(ds_obj, TRAIN_PATH_KEY).exists():
            raise ValueError(f"The train file for dataset {name} was not found (property '{TRAIN_PATH_KEY}')!")

    def _analyze_dataset(self, name: str, ds_obj: dict) -> CDEntry:
        dataset_id = _dataset_id(name)
        dataset_type = ds_obj.get(TYPE_KEY, "unknown")
        period = ds_obj.get(PERIOD_KEY, None)

        test_path = self._extract_path(ds_obj, TEST_PATH_KEY)
        train_path = None
        if TRAIN_PATH_KEY in ds_obj:
            train_path = self._extract_path(ds_obj, TRAIN_PATH_KEY)

        # get training type by inspecting training file
        training_type = _training_type(train_path)

        # analyze test time series
        dm = DatasetAnalyzer(dataset_id, is_train=False, dataset_path=test_path)

        return CDEntry(test_path, train_path, Dataset(
            datasetId=dataset_id,
            dataset_type=dataset_type,
            training_type=training_type,
            dimensions=dm.metadata.dimensions,
            length=dm.metadata.length,
            contamination=dm.metadata.contamination,
            min_anomaly_length=dm.metadata.anomaly_length.min,
            median_anomaly_length=dm.metadata.anomaly_length.median,
            max_anomaly_length=dm.metadata.anomaly_length.max,
            num_anomalies=dm.metadata.num_anomalies,
            period_size=period
        ))

    def get_collection_names(self) -> List[str]:
        return ["custom"]

    def get_dataset_names(self) -> List[str]:
        return [name for name in self._dataset_store]

    def get_path(self, dataset_name: str, train: bool) -> Path:
        dataset = self._dataset_store[dataset_name]

        if train:
            train_path = dataset.train_path
            if train_path is None:
                raise ValueError(f"Custom dataset {dataset_name} is unsupervised and has no training time series!")
            else:
                return train_path

        return dataset.test_path

    def get(self, dataset_name: str) -> Dataset:
        return self._dataset_store[dataset_name].details

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
        if (collection is not None and collection not in self.get_collection_names()) or (
                dataset is not None and dataset not in self.get_dataset_names()):
            return []
        else:
            selectors = []
            # used for an early-skip already
            # if dataset is not None:
            #     selectors.append(lambda meta: meta.datasetId[1] == dataset)
            if dataset_type is not None:
                selectors.append(lambda meta: meta.dataset_type == dataset_type)
            if datetime_index is not None:
                warnings.warn("Filter for index type (datetime or int) is not supported for custom dataset! "
                              "Ignoring it!")
            if training_type is not None:
                selectors.append(lambda meta: meta.training_type == training_type)
            if input_dimensionality is not None:
                selectors.append(lambda meta: meta.input_dimensionality == input_dimensionality)
            if min_anomalies is not None:
                selectors.append(lambda meta: meta.num_anomalies >= min_anomalies)
            if max_anomalies is not None:
                selectors.append(lambda meta: meta.num_anomalies <= max_anomalies)
            if max_contamination is not None:
                selectors.append(lambda meta: meta.contamination <= max_contamination)

            custom_datasets = []
            for d in self._dataset_store:
                if dataset is not None and dataset != d:
                    continue

                _, _, metadata = self._dataset_store[d]
                if np.all([fn(metadata) for fn in selectors]):
                    custom_datasets.append(d)
            return [("custom", name) for name in custom_datasets]
