import numpy as np
from enum import Enum
import json
from pathlib import Path
from typing import Tuple

from timeeval.utils.label_formatting import id2labels


class Datasets(Enum):
    SED = "sed"
    TAXI = "taxi"

    def load(self, dataset_config: Path) -> np.ndarray:
        dataset_store = json.load(dataset_config.open("r"))
        data_file = dataset_store[self.value]["data"]
        label_file = dataset_store[self.value]["labels"]
        data = np.loadtxt(data_file)
        labels = np.loadtxt(label_file, dtype=np.long).reshape(-1)
        if data.shape[0] != labels.shape[0]:
            labels = id2labels(labels, data.shape[0])
        return np.column_stack((data, labels))

    def get_path(self, dataset_config: Path) -> Tuple[Path, Path]:
        dataset_store = json.load(dataset_config.open("r"))
        data_file = dataset_store[self.value]["data"]
        label_file = dataset_store[self.value]["labels"]
        return Path(data_file), Path(label_file)
