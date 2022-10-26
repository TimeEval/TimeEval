import argparse

import numpy as np
import pandas as pd

from timeeval.metrics import DefaultMetrics, RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS, PrecisionAtK, FScoreAtK


all_metrics = {
    DefaultMetrics.PR_AUC.name: DefaultMetrics.PR_AUC,
    DefaultMetrics.ROC_AUC.name: DefaultMetrics.ROC_AUC,
    "RANGE_P_RANGE_R_AUC": DefaultMetrics.RANGE_PR_AUC,
    DefaultMetrics.FIXED_RANGE_PR_AUC.name: DefaultMetrics.FIXED_RANGE_PR_AUC,
    DefaultMetrics.AVERAGE_PRECISION.name: DefaultMetrics.AVERAGE_PRECISION,
    "PRECISION_AT_K": PrecisionAtK(),
    "F_SCORE_AT_K": FScoreAtK(),
    "RANGE_PR_AUC": RangePrAUC(),
    "RANGE_PR_AUC_COMP": RangePrAUC(compatibility_mode=True),
    "RANGE_ROC_AUC": RangeRocAUC(),
    "RANGE_ROC_AUC_COMP": RangeRocAUC(compatibility_mode=True),
    "RANGE_PR_VUS": RangePrVUS(),
    "RANGE_PR_VUS_COMP": RangePrVUS(compatibility_mode=True),
    "RANGE_ROC_VUS": RangeRocVUS(),
    "RANGE_ROC_VUS_COMP": RangeRocVUS(compatibility_mode=True),
}


def _create_arg_parser():
    parser = argparse.ArgumentParser(description=f"Metric calculation and plotting")

    parser.add_argument("input_file", type=str, help="Path to input file with the anomaly scores (point-based)")
    parser.add_argument("label_file", type=str, help="Path to the label file (looks for 'is_anomaly' or "
                                                     "'AnomalyLabel' column, otherwise takes the last column)")
    parser.add_argument("--metric", type=str, default=DefaultMetrics.ROC_AUC.name,
                        choices=list(all_metrics.keys()),
                        help="Metric to compute")

    return parser


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()
    metric: str = args.metric.upper().replace("-", "_")

    anomaly_scores = pd.read_csv(args.input_file, header=None).iloc[:, 0].values
    df = pd.read_csv(args.label_file)
    if "is_anomaly" in df.columns:
        anomaly_labels = df["is_anomaly"].values
    elif "AnomalyLabel" in df.columns:
        anomaly_labels = df["AnomalyLabel"].values
    else:
        anomaly_labels = df.iloc[:, -1].values
    anomaly_labels = anomaly_labels.astype(np.int_)

    print(f"Anomaly labels: {anomaly_labels.shape}")
    print(f"Anomaly scores: {anomaly_scores.shape}")

    result = all_metrics[metric](anomaly_labels, anomaly_scores)
    print(f"\n{metric} score = {result}")
