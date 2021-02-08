from pathlib import Path

import numpy as np
import pandas as pd


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    return df.values[:, -1]


def extract_features(df: pd.DataFrame) -> np.ndarray:
    return df.values[:, 1:-1]


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)
