import argparse
from pathlib import Path

from timeeval import Metric
from latex_generator.latex_plot_generator import LatexPlotGenerator


def _register_latexplotgenerator_parser(subparsers) -> argparse.ArgumentParser:
    all_metric_choices = [Metric.ROC_AUC.name,
                          Metric.PR_AUC.name,
                          Metric.RANGE_PR_AUC.name,
                          Metric.AVERAGE_PRECISION.name]
    parser: argparse.ArgumentParser = subparsers.add_parser("overview-table",
        description="Generates the main LaTeX table containing the TSAD algorithms and their different metric boxplots. "
                    "The table uses the booktabs package and tikz/pgfplots to generate the metric boxplots."
    )
    parser.add_argument("--metrics", type=str, nargs="*", default=all_metric_choices[:2], choices=all_metric_choices,
                        help="Metrics to consider in the overview. (default: %(default)s)")
    parser.add_argument("-g", "--include-gutentag", action="store_true",
                        help="Include the ROC_AUC boxplots computed just on the GutenTAG datasets in the last column.")
    parser.add_argument("-r", "--relative", action="store_true",
                        help="Display the error, timeout, and OOMs relative to the overall number of runs.")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "max", "min"],
                        help="Aggregation used to sort the algorithms. (default: %(default)s)")
    parser.add_argument("--dominant-metric", type=str, default=Metric.ROC_AUC.name, choices=all_metric_choices,
                        help="Metric used to sort the algorithms. (default: %(default)s)")
    return parser


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Script to generate the plots for our paper on benchmarking time "
                                                 "series anomaly detection algorithms. Use the subcommands to generate "
                                                 "the desired paper plot.")
    subparsers = parser.add_subparsers(dest="command", required=True, title="command",
                                       description="Target plot to generate. Those commands have their own arguments!")
    parser.add_argument("--result-path", type=Path, required=True,
                        help="Path to the preprocessed results CSV file (must include method families; must exclude "
                             "filtered out datasets and algorithms).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Path/filename, where the plot/code should be written to.")

    _register_latexplotgenerator_parser(subparsers)
    return parser


def main():
    parser = _create_arg_parser()
    args: argparse.Namespace = parser.parse_args()
    if args.command == "overview-table":
        selected_metrics = args.metrics
        # selected_metrics = [Metric[m] for m in selected_metrics]
        dominant_metric = args.dominant_metric
        # dominant_metric = Metric[dominant_metric]
        gen = LatexPlotGenerator(args.result_path,
                                 dominant_quality_metric=dominant_metric,
                                 aggregation=args.aggregation)
        gen.generate_overview_table(args.out,
                                    metrics=selected_metrics,
                                    relative_rates=args.relative,
                                    include_gutentag_only=args.include_gutentag)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
