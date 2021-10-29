import argparse
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from timeeval.utils.label_formatting import id2labels


def process(data_file: Union[Path, str], label_file: Union[Path, str], out_file: Union[Path, str],
            convert_datetime: bool, unit: str) -> None:
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=np.uint).reshape(-1)
    if data.shape[0] != labels.shape[0]:
        labels = id2labels(labels, data.shape[0])

    index = pd.Series(np.arange(len(labels)))
    df = pd.DataFrame(index=index)
    df.index.name = "timestamp"
    if convert_datetime:
        df.index = pd.to_datetime(df.index, unit=unit)

    if len(data.shape) == 2 and data.shape[1] > 1:
        for dim in range(data.shape[1]):
            df[f"value_{dim}"] = data[:, dim]
    else:
        df["value"] = data
    df["is_anomaly"] = labels
    print(f"Writing preprocessed dataset to {out_file}")
    df.to_csv(out_file)


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate dataset")
    parser.add_argument("--data", type=Path, required=True, help="File path for dataset")
    parser.add_argument("--labels", type=Path, required=True,
                        help="File path for labels (either with the same shape or a list of indices")
    parser.add_argument("--out", type=Path, required=True, help="Output file path for the processed dataset")
    parser.add_argument("-d", "--convert-to-datetime", action="store_true", help="Convert indices to proper datetime")
    parser.add_argument("-u", "--datetime-unit", type=str, default="ms", help="Unit for datetime conversion")
    return parser.parse_args()


def main():
    args = _create_arg_parser()
    process(args.data, args.labels, args.out, args.convert_to_datetime, args.datetime_unit)


if __name__ == "__main__":
    main()
