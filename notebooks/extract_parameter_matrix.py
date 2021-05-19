import json
import pandas as pd
import argparse
from pathlib import Path


def _validate_directory(path: str) -> Path:
    path = Path(path)
    if path.is_dir():
        return path
    if not path.exists():
        raise ValueError(f"'{path}' does not exist!")
    raise ValueError(f"Argument '--algorithms-directory' must be a directory. '{path}' is not a directory!")


def _create_arg_parser():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        "--algorithms-directory",
        type=_validate_directory,
        default=_validate_directory("../../timeeval-algorithms/"),
        required=False
    )

    argument_parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("../parameter-matrix.csv"),
        required=False
    )

    argument_parser.add_argument(
        "--details-path",
        type=Path,
        required=False
    )

    return argument_parser


def extract_parameter_matrix(args: argparse.Namespace):
    dirs: Path = args.algorithms_directory
    global_parameters = pd.DataFrame()

    for manifest_path in dirs.glob("*/manifest.json"):
        manifest = json.load(manifest_path.open(mode="r"))
        title = manifest.get("title")
        # training parameters
        training_parameters = pd.DataFrame(manifest.get("trainingStep", {}).get("parameters", []))
        training_parameters["title"] = title
        training_parameters["executionType"] = "training"

        # execution parameters
        execution_parameters = pd.DataFrame(manifest.get("executionStep", {}).get("parameters", []))
        execution_parameters["title"] = title
        execution_parameters["executionType"] = "execution"

        global_parameters = global_parameters.append(training_parameters, ignore_index=True)\
                                             .append(execution_parameters, ignore_index=True)

    global_parameters["type"] = global_parameters["type"].apply(lambda x: x.lower().strip())

    if "details_path" in args:
        global_parameters.to_csv(args.details_path, index=False)

    def combine_execution_types(x):
        if len(x["executionType"]) > 1:
            return x[x["executionType"] == "training"]["defaultValue"].values.reshape(-1)[0]
        return x["defaultValue"].values.reshape(-1)[0]

    parameter_matrix = global_parameters.groupby(["title", "name", "type"]).apply(combine_execution_types).unstack(level=[1,2])

    # remove columns that are completely empty
    parameter_matrix = parameter_matrix.loc[:, parameter_matrix.columns[~parameter_matrix.isna().all()]]


    # sort columns by frequent usages across algorithms
    column_sorting = parameter_matrix.count().values.argsort()[::-1]
    parameter_matrix.iloc[:, column_sorting].to_csv(args.output_path)


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()
    extract_parameter_matrix(args)
