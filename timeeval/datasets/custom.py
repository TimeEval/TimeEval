import json
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
import pandas as pd

from timeeval.datasets.custom_base import CustomDatasetsBase
from timeeval.utils.label_formatting import id2labels


class CustomDatasets(CustomDatasetsBase):
    _dataset_store: dict

    def __init__(self, dataset_config: Union[str, Path]):
        super().__init__()
        with Path(dataset_config).open("r") as f:
            self._dataset_store = json.load(f)

    def _validate_dataset(self, ds_obj: dict) -> bool:
        return "train_type" in ds_obj and "dataset" in ds_obj

    def _load_one_file(self, ds_obj: dict) -> pd.DataFrame:
        dataset_file = ds_obj["dataset"]
        df = pd.read_csv(dataset_file)
        return df

    def is_loaded(self):
        return True

    def get_dataset_names(self) -> List[str]:
        return [name for name in self._dataset_store]

    def load_df(self, dataset_name: str) -> pd.DataFrame:
        ds_obj = self._dataset_store[dataset_name]

        if self._validate_dataset(ds_obj):
            return self._load_one_file(ds_obj)
        else:
            raise ValueError(
                "A dataset obj in your dataset config file must have either 'data' and 'labels' paths or one 'dataset' path.")

    def get_path(self, dataset_name: str) -> Tuple[Path, Optional[Path]]:
        ds_obj = self._dataset_store[dataset_name]

        if self._validate_dataset(ds_obj):
            if self._is_one_file(ds_obj):
                return Path(ds_obj.get("dataset")), None
            else:
                data_file = ds_obj.get("data")
                labels_file = ds_obj.get("labels")
                return Path(data_file), Path(labels_file)
        else:
            raise ValueError(
                "A dataset obj in your dataset config file must have either 'data' and 'labels' paths or one 'dataset' path.")
