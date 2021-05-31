import argparse
from pathlib import Path

from .codegen import AlgorithmGenerator


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate python stubs for algorithms of the timeeval-algorithms "
                                                 "repository that are implemented as Docker images.")
    parser.add_argument("repository_path", type=Path,
                        help="Folder containing the timeeval-algorithms repository (source for the Docker images; "
                             "required structure: each algorithm in its own folder, the folder contains a "
                             "manifest.yaml and a README.md file, and the folder name must be a valid python-identifier.")
    parser.add_argument("-f", "--force", action="store_true", default=False,
                        help="Force creation of folders and files. Will overwrite existing files.")
    parser.add_argument("-s", "--skip-pull", action="store_true", default=False,
                        help="Sets default Docker image pull policy for all algorithms. Can be overridden during usage")
    parser.add_argument("-t", "--timeout", type=str, default="1 hour",
                        help="Sets default timeout for Docker run of all algorithms. Can be overridden during usage")
    return parser.parse_args()


args = _create_arg_parser()
print(f"#### Reading algorithm metadata from {args.repository_path}")
generator = AlgorithmGenerator(
    timeeval_algorithms=args.repository_path,
    skip_pull=args.skip_pull,
    default_timeout=args.timeout,
)

target_dir = Path(__file__).parent.parent / "algorithms"
if not target_dir.is_dir():
    target_dir.mkdir(exist_ok=True)

print(f"\n#### Generating algorithm stubs in {target_dir}")
generator.generate_all(target=target_dir, force=args.force)
print("#### DONE")
