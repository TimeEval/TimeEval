import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def concat_dataset(data: np.ndarray, labels: np.ndarray, reps: int) -> Tuple[np.ndarray, np.ndarray]:
    data_size = len(data)
    label_size = len(labels)

    new_data = np.tile(data, reps=reps)
    new_labels = np.tile(labels, reps=reps)
    if data_size > label_size:
        for i in range(reps):
            new_labels[i * label_size:(i * label_size) + label_size] += data_size * i

    return new_data, new_labels


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate dataset")
    parser.add_argument("--dataset-file", type=Path, required=True, help="File path for dataset")
    parser.add_argument("--labels-file", type=Path, required=True, help="File path for labels")
    parser.add_argument("--t", type=int, required=True, help="Concatenation factor")
    return parser.parse_args()


def main():
    args = _create_arg_parser()
    data = np.loadtxt(args.dataset_file)
    labels = np.loadtxt(args.labels_file).reshape(-1)

    new_data, new_labels = concat_dataset(data, labels, args.t)

    new_data.tofile(str(args.dataset_file) + f".{args.t}", sep="\n")
    new_labels.tofile(str(args.labels_file) + f".{args.t}", sep="\n")


if __name__ == "__main__":
    main()
