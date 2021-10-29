import argparse
import logging
from pathlib import Path

from timeeval.constants import RESULTS_CSV


class ResultSummary:
    KEY_COLUMNS = ["algorithm", "collection", "dataset", "repetition", "hyper_params_id"]

    def __init__(self, result_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.result_folder = result_folder

    def _inspect_result_folder(self):
        results_csv = self.result_folder / RESULTS_CSV
        if results_csv.exists():
            self._logger.warning(f"There already exists a {RESULTS_CSV} file, backing it up!")
            results_csv.rename(results_csv.parent / "results.bak.csv")

    def create(self):
        pass


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Takes an experiment result folder and re-creates the results.csv-file from the experiment backups."
    )
    parser.add_argument("result_folder", type=Path, help="Folder of the experiment")
    parser.add_argument("--loglevel", default="INFO", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help="Set logging verbosity (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _create_arg_parser()
    logging.basicConfig(level=args.loglevel)
    rs = ResultSummary(args.result_folder)
    rs.create()
