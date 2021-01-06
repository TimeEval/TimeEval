import numpy as np
from pathlib import Path
import argparse


def create_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate dataset")
    parser.add_argument("--dataset-file", type=Path, required=True, help="File path for dataset")
    parser.add_argument("--labels-file", type=Path, required=True, help="File path for labels")
    parser.add_argument("--t", type=int, required=True, help="Concatenation factor")
    return parser.parse_args()


def main():
    args = create_parser()
    data = np.loadtxt(args.dataset_file)
    labels = np.loadtxt(args.labels_file).reshape(-1)

    data_size = len(data)
    label_size = len(labels)

    new_data = np.tile(data, reps=args.t)
    new_labels = np.tile(labels, reps=args.t)
    if data_size > label_size:
        for i in range(args.t):
            new_labels[i*label_size:(i*label_size)+label_size] += data_size*i

    new_data.tofile(str(args.dataset_file)+f".{args.t}", sep="\n")
    new_labels.tofile(str(args.labels_file)+f".{args.t}", sep="\n")


if __name__ == "__main__":
    main()
