import json
from pathlib import Path
from typing import List, Union

from timeeval.datasets.custom_base import CustomDatasetsBase


class CustomDatasets(CustomDatasetsBase):

    def __init__(self, dataset_config: Union[str, Path]):
        super().__init__()
        with Path(dataset_config).open("r") as f:
            store = json.load(f)
        for dataset in store:
            if not self._validate_dataset(store[dataset]):
                raise ValueError(
                    "A dataset obj in your dataset config file must have 'dataset' and 'train_type' paths.")

        self._dataset_store = store

    def _validate_dataset(self, ds_obj: dict) -> bool:
        return "train_type" in ds_obj and "dataset" in ds_obj

    def get_collection_names(self) -> List[str]:
        return ["custom"]

    def get_dataset_names(self) -> List[str]:
        return [name for name in self._dataset_store]

    def get_path(self, dataset_name: str, train: bool) -> Path:
        ds_obj = self._dataset_store[dataset_name]

        train_type = ds_obj["train_type"]
        if train and train_type != "train":
            raise ValueError(f"Custom dataset {dataset_name} is meant for testing and not for training!")
        if not train and train_type != "test":
            raise ValueError(f"Custom dataset {dataset_name} is meant for training and not for testing!")

        return Path(ds_obj.get("dataset"))
