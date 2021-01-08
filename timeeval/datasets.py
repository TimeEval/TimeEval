import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Optional
import datetime as dt

from timeeval.utils.label_formatting import id2labels


class Datasets:
    def __init__(self, value: str):
        self.value = value

    def _is_one_file(self, ds_obj: dict) -> bool:
        return "dataset" in ds_obj

    def _validate_dataset(self, ds_obj: dict) -> bool:
        return ("data" in ds_obj and "labels" in ds_obj) or "dataset" in ds_obj

    def _load_one_file(self, ds_obj: dict) -> pd.DataFrame:
        dataset_file = ds_obj["dataset"]
        df = pd.read_csv(dataset_file)
        return df

    def _load_two_files(self, ds_obj: dict) -> pd.DataFrame:
        data_file = ds_obj["data"]
        label_file = ds_obj["labels"]

        data = np.loadtxt(data_file)
        labels = np.loadtxt(label_file, dtype=np.long).reshape(-1)
        if data.shape[0] != labels.shape[0]:
            labels = id2labels(labels, data.shape[0])

        df = pd.DataFrame()
        df["timestamp"] = [dt.datetime(year=1970, month=1, day=1) + dt.timedelta(milliseconds=int(x)) for x in np.arange(len(labels))]

        if len(data.shape) == 2 and data.shape[1] > 1:
            for dim in range(data.shape[1]):
                df[f"value_{dim}"] = data[:, dim]
        else:
            df["value"] = data
        df["is_anomaly"] = labels
        return df

    def load(self, dataset_config: Path) -> pd.DataFrame:
        dataset_store = json.load(dataset_config.open("r"))
        ds_obj = dataset_store[self.value]

        if self._validate_dataset(ds_obj):
            if self._is_one_file(ds_obj):
                return self._load_one_file(ds_obj)
            else:
                return self._load_two_files(ds_obj)
        else:
            raise ValueError("A dataset obj in your dataset config file must have either 'data' and 'labels' paths or one 'dataset' path.")

    def get_path(self, dataset_config: Path) -> Tuple[Path, Optional[Path]]:
        dataset_store = json.load(dataset_config.open("r"))
        ds_obj = dataset_store[self.value]

        if self._validate_dataset(ds_obj):
            if self._is_one_file(ds_obj):
                return Path(ds_obj.get("dataset")), None
            else:
                data_file = ds_obj.get("data")
                labels_file = ds_obj.get("labels")
                return Path(data_file), Path(labels_file)
