import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import json

from timeeval import TimeEval, Algorithm
from timeeval.constants import RESULTS_CSV, HYPER_PARAMETERS, METRICS_CSV

# required to build a lookup-table for algorithm implementations
import timeeval_experiments.algorithms as algorithms
# noinspection PyUnresolvedReferences
from timeeval_experiments.algorithms import *


@dataclass
class Experiment:
    algorithm: Algorithm
    collection_name: str
    dataset_name: str
    repetition: int
    dataset_training_type: str
    dataset_input_dimensionality: str
    hyper_params: dict
    hyper_params_id: str
    metrics: pd.DataFrame


class ResultSummary:

    def __init__(self, results_path: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.results_path = results_path
        self.algos = self._build_algorithm_dict()

    @staticmethod
    def _build_algorithm_dict() -> Dict[str, Algorithm]:
        algo_names = [a for a in dir(algorithms) if not a.startswith("__")]
        algo_list: List[Algorithm] = [eval(f"{a}()") for a in algo_names]
        algos: Dict[str, Algorithm] = {}
        for a in algo_list:
            algos[a.name] = a
        return algos

    @staticmethod
    def _load_hyper_params(path: Path) -> Dict[str, Any]:
        hp_path = path / HYPER_PARAMETERS
        try:
            with hp_path.open("r") as fh:
                params = json.load(fh)
            return params
        except FileNotFoundError as e:
            # deal with error
            return {}

    @staticmethod
    def _load_metrics(path: Path) -> pd.DataFrame:
        m_path = path / METRICS_CSV
        return pd.read_csv(m_path)

    def _inspect_result_folder(self) -> None:
        results_csv = self.results_path / RESULTS_CSV
        if results_csv.exists():
            self._logger.warning(f"There already exists a {RESULTS_CSV} file, backing it up!")
            results_csv.rename(results_csv.parent / "results.bak.csv")

    def create(self):
        self._logger.info(f"Reading results from the directory {self.results_path}")
        self._inspect_result_folder()

        experiments: List[Experiment] = []
        for d_a in self.results_path.iterdir():
            if d_a.is_dir():
                algorithm_name = d_a.name
                for d_hpi in d_a.iterdir():
                    hyper_params_id = d_hpi.name
                    for d_c in d_hpi.iterdir():
                        collection_name = d_c.name
                        for d_d in d_c.iterdir():
                            dataset_name = d_d.name
                            for d_r in d_d.iterdir():
                                repetition = int(d_r.name)
                                # find algo
                                algo = self.algos[algorithm_name]
                                # read hyper parameters from file
                                hyper_params = self._load_hyper_params(d_r)
                                # read metrics from file
                                metrics = self._load_metrics(d_r)
                                # find out dataset input dimensionality and training type
                                dataset_input_dimensionality = ""
                                dataset_training_type = ""

                                experiments.append(Experiment(
                                    algorithm=algo,
                                    collection_name=collection_name,
                                    dataset_name=dataset_name,
                                    repetition=repetition,
                                    hyper_params=hyper_params,
                                    hyper_params_id=hyper_params_id,
                                    metrics=metrics,
                                    dataset_input_dimensionality=dataset_input_dimensionality,
                                    dataset_training_type=dataset_training_type
                                ))

        print(experiments)
        return
        metric_names = []
        df = pd.DataFrame(columns=TimeEval.RESULT_KEYS + metric_names)


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
