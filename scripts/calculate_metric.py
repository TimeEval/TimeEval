import argparse

import numpy as np
import pandas as pd

from timeeval import Metric


def _create_arg_parser():
    parser = argparse.ArgumentParser(description=f"Metric calculation and plotting")

    parser.add_argument("input_file", type=str, help="Path to input file with the anomaly scores (point-based)")
    parser.add_argument("label_file", type=str, help="Path to the label file (looks for 'is_anomaly' or "
                                                     "'AnomalyLabel' column, otherwise takes the last column)")
    parser.add_argument("--metric", type=str, default=Metric.ROC_AUC.name,
                        choices=[
                            Metric.ROC_AUC.name, Metric.PR_AUC.name,
                            Metric.RANGE_PR_AUC.name, Metric.AVERAGE_PRECISION.name
                        ],
                        help="Metric to compute")
    parser.add_argument("-p", "--plot", action="store_true", help="Enable plotting the metric curve")
    parser.add_argument("--save-plot", action="store_true", help="Save the metric plot as pdf")

    return parser


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()
    if args.metric == Metric.AVERAGE_PRECISION.name and args.plot is True:
        print("Cannot create a curve plot for the average precision metric!")
        args.plot = False

    anomaly_scores = pd.read_csv(args.input_file, header=None).iloc[:, 0].values
    df = pd.read_csv(args.label_file)
    if "is_anomaly" in df.columns:
        anomaly_labels = df["is_anomaly"].values
    elif "AnomalyLabel" in df.columns:
        anomaly_labels = df["AnomalyLabel"].values
    else:
        anomaly_labels = df.iloc[:, -1].values
    anomaly_labels = anomaly_labels.astype(np.int_)

    print(f"Anomaly labels: {anomaly_labels}")
    print(f"Anomaly scores: {anomaly_scores}")

    result = Metric[args.metric](anomaly_labels, anomaly_scores, plot=args.plot, plot_store=args.save_plot)
    print(f"\n{args.metric} score = {result}")
