import argparse
from pathlib import Path
import sys
from typing import Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeeval.constants import ANOMALY_SCORES_TS, RESULTS_CSV
from timeeval.datasets.datasets import Datasets
from timeeval.utils.results_path import generate_experiment_path


class Logger:
    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.current_step = 0

    def log(self, text: str):
        self.current_step += 1
        print(f"[{self.current_step}/{self.n_steps}]\t{text}")


logger = Logger(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, help="Path to the results folder")
    parser.add_argument("--id", type=int, help="Row number in results.csv")
    parser.add_argument("--datasets", type=Path, help="Path to datasets folder")
    parser.add_argument("--pylustrator", action="store_true", help="Use pylustrator for plotting!")
    args = parser.parse_args(sys.argv[1:])
    return args


def load_results(args: argparse.Namespace) -> pd.DataFrame:
    return pd.read_csv(args.path / RESULTS_CSV)


def get_experiment_row(args: argparse.Namespace) -> pd.DataFrame:
    df = load_results(args)
    return df.iloc[args.id]


def get_experiment_path(args: argparse.Namespace) -> Path:
    experiment_row = get_experiment_row(args)
    return generate_experiment_path(args.path,
                                    experiment_row.algorithm,
                                    experiment_row.hyper_params_id,
                                    experiment_row.collection,
                                    experiment_row.dataset,
                                    experiment_row.repetition)


def get_anomaly_scores(args: argparse.Namespace) -> np.ndarray:
    logger.log("Loading Anomaly Scores")
    directory = get_experiment_path(args)
    return np.genfromtxt(directory / ANOMALY_SCORES_TS)


def get_dataset(args: argparse) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    experiment_row = get_experiment_row(args)
    logger.log(f"Loading Dataset {experiment_row.dataset} from {experiment_row.collection}")
    dm = Datasets(args.datasets)
    df = dm.get_dataset_df((experiment_row.collection, experiment_row.dataset), train=False)

    column_names = df.columns[1:-1]
    values = df.values[:, 1:-1]
    labels = df.values[:, -1]
    return values, column_names, labels


def plot_dataset(args: argparse.Namespace) -> List[Any]:
    values, column_names, labels = get_dataset(args)
    rows = len(column_names) + 2  # column_names + labels + scores

    fig, axs = plt.subplots(rows, 1, sharex=True)

    logger.log("Plotting Dataset Channels")
    for c in range(values.shape[1]):
        ax = axs[c]
        ax.set_title(column_names[c])
        ax.plot(values[:, c])

    ax = axs[-2]
    ax.set_title("is anomaly")
    ax.plot(labels)

    return axs


def plot_anomaly_scores(args: argparse.Namespace, axs):
    scores = get_anomaly_scores(args)
    logger.log("Plotting Anomaly Scores")
    ax = axs[-1]
    ax.set_title("anomaly scores")
    ax.plot(scores)


def main():
    args = parse_args()

    if args.pylustrator:
        import pylustrator
        pylustrator.start()

    axs = plot_dataset(args)
    plot_anomaly_scores(args, axs)

    logger.log("Showing Plots")
    plt.show()


if __name__ == "__main__":
    main()
