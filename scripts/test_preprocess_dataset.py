import filecmp
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess_dataset import process

data_file = "./tests/example_data/data.txt"
label_file = "./tests/example_data/labels.txt"
index_label_file = "./tests/example_data/labels.indices.txt"


def test_process(tmp_path: Path) -> None:
    target = tmp_path / "test.csv"
    process(data_file, label_file, target, convert_datetime=False, unit="ms")
    filecmp.cmp(target, "./tests/example_data/dataset.train.csv")


def test_process_indices(tmp_path: Path) -> None:
    target = tmp_path / "test.csv"
    process(data_file, index_label_file, target, convert_datetime=False, unit="ms")
    df = pd.read_csv(target)
    indices = np.loadtxt(index_label_file)
    assert np.all(df.loc[indices, "is_anomaly"])
    other = ~df.index.isin(list(indices if isinstance(indices, list) else [indices]))
    assert not np.all(df.loc[other, "is_anomaly"])


def test_process_convert_datetime(tmp_path: Path) -> None:
    target = tmp_path / "test.csv"
    process(data_file, label_file, target, convert_datetime=True, unit="s")
    df = pd.read_csv(target, parse_dates=["timestamp"], infer_datetime_format=True)
    assert df["timestamp"].dtype == np.dtype("<M8[ns]")
