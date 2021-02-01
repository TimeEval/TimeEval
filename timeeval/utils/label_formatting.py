import numpy as np


def id2labels(ids: np.ndarray, data_length: int) -> np.ndarray:
    labels = np.zeros(data_length, dtype=np.int64)
    labels[ids] = 1
    return labels


def labels2id(labels: np.ndarray) -> np.ndarray:
    ids = labels[labels == 1]
    return ids
