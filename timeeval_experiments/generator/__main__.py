import argparse
import sys
from pathlib import Path
from typing import Any, List

from .codegen import AlgorithmGenerator
from .param_config_gen import ParamConfigGenerator

ALGO_STUBS_COMMAND = "algo-stubs"
PARAM_CONFIG_COMMAND = "param-config"


def _create_generate_algorithm_stubs_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(ALGO_STUBS_COMMAND,
                                   help="Generate python stubs for algorithms of the timeeval-algorithms "
                                        "repository that are implemented as Docker images.")
    parser.add_argument("repository_path", type=Path,
                        help="Folder containing the timeeval-algorithms repository (source for the Docker images; "
                             "required structure: each algorithm in its own folder, the folder contains a "
                             "manifest.yaml and a README.md file, and the folder name must be a valid python-identifier.")
    parser.add_argument("-f", "--force", action="store_true", default=False,
                        help="Force creation of folders and files. Will overwrite existing files.")
    parser.add_argument("-s", "--skip-pull", action="store_true", default=False,
                        help="Sets default Docker image pull policy for all algorithms. Can be overridden during usage")
    return parser


def _create_generate_param_config_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(PARAM_CONFIG_COMMAND,
                                   help="Generate parameter configuration (JSON) based on parameter matrix.")
    parser.add_argument("parameter_matrix_path", type=Path,
                        help="Path to the file that contains the parameter matrix (direct CSV export of the Google "
                             "Spreadsheet document).")
    parser.add_argument("-o", "--output", type=Path, default="param-config.json", required=False,
                        help="Output filename.")
    parser.add_argument("-f", "--force", action="store_true", default=False,
                        help="Will overwrite existing entries in the parameter configuration file. If `False` "
                             f"(default) only the keys {ParamConfigGenerator.FIXED_KEY}, "
                             f"{ParamConfigGenerator.SHARED_KEY}, {ParamConfigGenerator.DEPENDENT_KEY}, and "
                             f"{ParamConfigGenerator.OPTIMIZED_KEY} are updated. Other keys are untouched")
    return parser


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate various configuration files and python stubs for TimeEval "
                                                 "algorithms of the timeeval-algorithms repository. See subcommands "
                                                 "for further details.")
    subparsers = parser.add_subparsers(dest="command")
    _create_generate_algorithm_stubs_parser(subparsers)
    _create_generate_param_config_parser(subparsers)

    return parser


def _run_algorithm_gen(args: argparse.Namespace) -> None:
    print(f"#### Reading algorithm metadata from {args.repository_path}")
    generator = AlgorithmGenerator(
        timeeval_algorithms=args.repository_path,
        skip_pull=args.skip_pull,
    )

    target_dir = Path(__file__).parent.parent / "algorithms"
    if not target_dir.is_dir():
        target_dir.mkdir(exist_ok=True)

    print(f"\n#### Generating algorithm stubs in {target_dir}")
    generator.generate_all(target=target_dir, force=args.force)
    print("#### DONE")


def _run_config_gen(args: argparse.Namespace) -> None:
    print(f"#### Reading parameter matrix from {args.parameter_matrix_path}")
    generator = ParamConfigGenerator(matrix_path=args.parameter_matrix_path)

    target: Path = args.output
    print(f"\n#### Generating parameter config at {target}")
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    generator.generate(
        target=args.output,
        overwrite=args.force
    )
    pass


def main(argv: List[str]):
    parser = _create_arg_parser()
    args = parser.parse_args(argv[1:])
    if not args.command:
        parser.print_help()
        sys.exit(0)
    elif args.command == ALGO_STUBS_COMMAND:
        _run_algorithm_gen(args)
    elif args.command == PARAM_CONFIG_COMMAND:
        _run_config_gen(args)
    else:
        print(f"Command {args.command} not found!")
        parser.print_usage()
        sys.exit(1)


main(sys.argv)
