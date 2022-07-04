from pathlib import Path

import numpy as np
import pandas as pd


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    labels: np.ndarray = df.values[:, -1].astype(np.float64)
    return labels


def extract_features(df: pd.DataFrame) -> np.ndarray:
    features: np.ndarray = df.values[:, 1:-1]
    return features


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True)


def load_labels_only(path: Path) -> np.ndarray:
    labels: np.ndarray = pd.read_csv(path, usecols=["is_anomaly"])["is_anomaly"].values.astype(np.float64)
    return labels
