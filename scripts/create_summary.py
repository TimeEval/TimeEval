import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# required to build a lookup-table for algorithm implementations
import timeeval_experiments.algorithms as algorithms
from timeeval import Algorithm, Status, Datasets, Metric
from timeeval.data_types import ExecutionType
from timeeval.experiments import Experiment as TimeEvalExperiment
from timeeval.adapters.docker import SCORES_FILE_NAME as DOCKER_SCORES_FILE_NAME
from timeeval.constants import RESULTS_CSV, HYPER_PARAMETERS, METRICS_CSV, EXECUTION_LOG, ANOMALY_SCORES_TS
# noinspection PyUnresolvedReferences
from timeeval.utils.datasets import load_labels_only
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines


@dataclass
class Experiment:
    path: Path
    algorithm: Algorithm
    collection_name: str
    dataset_name: str
    repetition: int
    dataset_training_type: str
    dataset_input_dimensionality: str
    hyper_params: Dict[str, Any]
    hyper_params_id: str
    metrics: Dict[str, float]
    status: Status

    @property
    def name(self) -> str:
        return f"{self.algorithm.name}-{self.dataset_name}-{self.hyper_params_id}-{self.repetition}"

    def to_dict(self):
        obj = {
            "algorithm": self.algorithm.name,
            "collection": self.collection_name,
            "dataset": self.dataset_name,
            "algo_training_type": self.algorithm.training_type.value,
            "algo_input_dimensionality": self.algorithm.input_dimensionality.value,
            "dataset_training_type": self.dataset_training_type,
            "dataset_input_dimensionality": self.dataset_input_dimensionality,
            "status": self.status,
            "hyper_params_id": self.hyper_params_id,
            "repetition": self.repetition,
            "hyper_params": self.hyper_params
        }
        if self.metrics:
            obj.update(self.metrics)
        return obj


class ResultSummary:

    def __init__(self, results_path: Path, data_path: Path, metric_list: Optional[List[Metric]] = None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.results_path = results_path
        self.data_path = data_path
        self.dmgr = Datasets(data_path, create_if_missing=False)
        self.algos = self._build_algorithm_dict()
        self.metric_list: List[Metric] = metric_list or Metric.default_list()
        self.df: Optional[pd.DataFrame] = None

    @staticmethod
    def _names_from_path(path: Path) -> Tuple[str, str, str]:
        dataset = path.parent.name
        hpi = path.parent.parent.parent.name
        algorithm = path.parent.parent.parent.parent.name
        return algorithm, dataset, hpi

    @staticmethod
    def _build_algorithm_dict() -> Dict[str, Algorithm]:
        algo_names = [a for a in dir(algorithms) if not a.startswith("__")]
        algo_list: List[Algorithm] = [eval(f"{a}()") for a in algo_names]
        algos: Dict[str, Algorithm] = {}
        for a in algo_list:
            algos[a.name] = a
        # add baselines
        increasing_baseline = Baselines.increasing()
        algos[increasing_baseline.name] = increasing_baseline
        random_baseline = Baselines.random()
        algos[random_baseline.name] = random_baseline
        normal_baseline = Baselines.normal()
        algos[normal_baseline.name] = normal_baseline
        # aliases for some renamed algorithms:
        algos["Image-embedding-CAE"] = algos["ImageEmbeddingCAE"]
        algos["LTSM-VAE"] = algos["LSTM-VAE"]
        return algos

    @staticmethod
    def _timeout_in_logs(path: Path) -> bool:
        log_path = path / EXECUTION_LOG
        with log_path.open("r") as fh:
            logs: List[str] = fh.readlines()
        timeout_logs = list(filter(lambda log: "Container timeout after" in log and "disregard" not in log, logs))
        return len(timeout_logs) > 0

    def _get_dataset_metadata(self, collection: str, dataset: str) -> Tuple[str, str]:
        try:
            meta = self.dmgr.get(collection, dataset)
            return meta.training_type.value, meta.input_dimensionality.value
        except KeyError as e:
            self._logger.error(f"Metadata about dataset {collection}-{dataset} not found!", exc_info=e)
            return "", ""

    def _load_hyper_params(self, path: Path) -> Dict[str, Any]:
        hp_path = path / HYPER_PARAMETERS
        try:
            with hp_path.open("r") as fh:
                params = json.load(fh)
            return params
        except FileNotFoundError as e:
            algo, dataset, hpi = self._names_from_path(path)
            self._logger.debug(f"Could not load hyper parameters for {algo} on {dataset} ({hpi}).", exc_info=e)
            return {}

    def _load_metrics(self, path: Path) -> Dict[str, float]:
        m_path = path / METRICS_CSV
        try:
            return pd.read_csv(m_path).iloc[0, :].to_dict()
        except FileNotFoundError as e:
            algo, dataset, hpi = self._names_from_path(path)
            self._logger.debug(f"Could not load metrics for {algo} on {dataset} ({hpi}).", exc_info=e)
            return {}

    def _calculate_status(self, path: Path, metrics: Dict[str, float]) -> Status:
        # if scores and metrics --> ok
        status = Status.OK
        if not metrics:
            # if scores, but no metrics --> recalculate metrics
            if (path / DOCKER_SCORES_FILE_NAME).exists():
                algo, dataset, hpi = self._names_from_path(path)
                self._logger.warning(f"Found successful experiment with missing metrics: {algo}-{dataset} ({hpi})")
                raise ValueError("Successful experiment with missing metrics")

            # if no score and no metrics --> error or timeout
            # --> inspect execution.log, if timeout in logs --> timeout, else --> error
            elif self._timeout_in_logs(path):
                status = Status.TIMEOUT
            else:
                status = Status.ERROR
        return status

    def _update_metrics(self, exp: Experiment) -> None:
        self._logger.info(f"Re-calculating quality metrics for {exp.name}")
        y_scores = pd.read_csv(exp.path / DOCKER_SCORES_FILE_NAME).iloc[0, :].values
        if exp.algorithm.postprocess:
            param_grid = exp.algorithm.param_grid
            if len(param_grid) == 1:
                y_scores = exp.algorithm.postprocess(y_scores, {
                    "executionType": ExecutionType.EXECUTE,
                    "results_path": exp.path,
                    "hyper_params": param_grid[0],
                    "dataset_details": self.dmgr.get(exp.collection_name, exp.dataset_name)
                })
            else:
                self._logger.error(f"Cannot post-process algorithm scores!")
                return

        y_true = load_labels_only(self.dmgr.get_dataset_path((exp.collection_name, exp.dataset_name), train=False))
        y_true, y_scores = TimeEvalExperiment.scale_scores(y_true, y_scores)

        if not (exp.path / ANOMALY_SCORES_TS).exists():
            self._logger.warning(f"{exp.name}: Anomaly scores are missing, recreating!")
            # persist scores to disk
            y_scores.tofile(str(exp.path / ANOMALY_SCORES_TS), sep="\n")

        results = {}
        errors = 0
        for metric in self.metric_list:
            try:
                score = metric(y_true, y_scores)
                results[metric.name] = score
            except Exception as e:
                self._logger.debug(f"{exp.name}: Exception while computing metric {metric}!", exc_info=e)
                errors += 1
                continue

        # update metrics and write them to disk
        exp.metrics.update(results)
        if exp.metrics:
            self._logger.info(f"{exp.name}: Writing new metrics to {exp.path / METRICS_CSV}!")
            pd.DataFrame([exp.metrics]).to_csv(exp.path / METRICS_CSV, index=False)

        if errors == 0:
            exp.status = Status.OK
        else:
            self._logger.error(f"There were errors while computing metrics for {exp.name}. Please consult DEBUG logs!")

    def _inspect_result_folder(self) -> None:
        results_csv = self.results_path / RESULTS_CSV
        if results_csv.exists():
            self._logger.info(f"There already exists a {RESULTS_CSV} file, backing it up!")
            results_csv.rename(results_csv.parent / "results.bak.csv")

    def create(self, fix_metrics: bool = False):
        self._inspect_result_folder()

        self._logger.info(f"Reading results from the directory {self.results_path} and parsing them...")
        experiments: List[Experiment] = []
        to_fix_experiments: List[Experiment] = []
        for d_a in self.results_path.iterdir():
            if d_a.is_dir():
                algorithm_name = d_a.name
                self._logger.info(f"Parsing results of {algorithm_name}")
                # find algo
                try:
                    algo = self.algos[algorithm_name]
                except KeyError as e:
                    self._logger.error(f"Could not find metadata about algorithm '{algorithm_name}', skipping!",
                                       exc_info=e)
                    continue

                for d_hpi in d_a.iterdir():
                    hyper_params_id = d_hpi.name
                    for d_c in d_hpi.iterdir():
                        collection_name = d_c.name
                        for d_d in d_c.iterdir():
                            dataset_name = d_d.name
                            dataset_input_dimensionality, dataset_training_type = \
                                self._get_dataset_metadata(collection_name, dataset_name)
                            for d_r in d_d.iterdir():
                                repetition = int(d_r.name)
                                # read hyper parameters from file
                                hyper_params = self._load_hyper_params(d_r)
                                # read metrics from file
                                metrics = self._load_metrics(d_r)

                                # figure out status:
                                try:
                                    status = self._calculate_status(d_r, metrics)
                                except ValueError as e:
                                    if "Successful experiment with missing metrics" in str(e):
                                        to_fix_experiments.append(Experiment(
                                            path=d_r,
                                            algorithm=algo,
                                            collection_name=collection_name,
                                            dataset_name=dataset_name,
                                            repetition=repetition,
                                            hyper_params=hyper_params,
                                            hyper_params_id=hyper_params_id,
                                            metrics=metrics,
                                            dataset_input_dimensionality=dataset_input_dimensionality,
                                            dataset_training_type=dataset_training_type,
                                            status=Status.ERROR
                                        ))
                                        continue
                                    else:
                                        raise e

                                experiments.append(Experiment(
                                    path=d_r,
                                    algorithm=algo,
                                    collection_name=collection_name,
                                    dataset_name=dataset_name,
                                    repetition=repetition,
                                    hyper_params=hyper_params,
                                    hyper_params_id=hyper_params_id,
                                    metrics=metrics,
                                    dataset_input_dimensionality=dataset_input_dimensionality,
                                    dataset_training_type=dataset_training_type,
                                    status=status
                                ))
        self._logger.info("Finished parsing results.")

        if fix_metrics and len(to_fix_experiments):
            self._logger.info(f"\nEncountered {len(to_fix_experiments)} experiments with missing metrics, "
                              "recalculating quality measures. Runtimes are lost in all cases!")
            for exp in to_fix_experiments:
                self._update_metrics(exp)
                experiments.append(exp)
            del to_fix_experiments
            self._logger.info("Finished recalculating metrics.")

        data = [exp.to_dict() for exp in experiments]
        self.df = pd.DataFrame(data)
        self.df.sort_values(by=["algorithm", "collection", "dataset"], inplace=True)
        self._logger.info("Finished!")

    def save(self) -> None:
        if self.df is not None:
            self.df.to_csv(self.results_path / RESULTS_CSV, index=False)
            self._logger.info(f"Stored experiment summary at {self.results_path / RESULTS_CSV}")
        else:
            raise ValueError("No results available! Run ResultSummary#create() first.")


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Takes an experiment result folder and re-creates the results.csv-file from the experiment backups."
    )
    parser.add_argument("result_folder", type=Path, help="Folder of the experiment")
    parser.add_argument("data_folder", type=Path, help="Folder, where the datasets from the experiment are stored."
                                                       "This script loads the dataset metadata directly from the "
                                                       "dataset index (datasets.csv).")
    parser.add_argument("--loglevel", default="INFO", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help="Set logging verbosity (default: %(default)s)")
    parser.add_argument("-f", "--fix", action="store_true", help="If successful experiments with missing metrics are "
                                                                 "found, try to re-calculate those metrics based on "
                                                                 "the algorithm (docker) scores.")
    all_metric_choices = [
                            Metric.ROC_AUC.name, Metric.PR_AUC.name,
                            Metric.RANGE_PR_AUC.name, Metric.AVERAGE_PRECISION.name
                        ]
    parser.add_argument("--metrics", type=str, nargs="*", default=all_metric_choices, choices=all_metric_choices,
                        help="Metrics to re-calculate if --fix is given as well. (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _create_arg_parser()
    logging.basicConfig(level=args.loglevel)
    if args.fix:
        metrics = args.metrics
        metrics = [Metric[m] for m in metrics]
    else:
        metrics = []

    rs = ResultSummary(args.result_folder, args.data_folder, metrics)
    rs.create(fix_metrics=args.fix)
    rs.save()
