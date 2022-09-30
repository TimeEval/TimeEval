import argparse
import logging
import multiprocessing

import multiprocessing_logging
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from timeeval import Algorithm, Metric, DefaultMetrics, MultiDatasetManager
from timeeval.constants import RESULTS_CSV, METRICS_CSV, ANOMALY_SCORES_TS
from timeeval.metrics import FScoreAtK, PrecisionAtK
from timeeval.utils.datasets import load_labels_only
from timeeval.utils.tqdm_joblib import tqdm_joblib


# required to build a lookup-table for algorithm implementations
import timeeval_experiments.algorithms as algorithms
# noinspection PyUnresolvedReferences
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines

INITIAL_WAITING_SECONDS = 5


def init_logging():
    """Workaround for logging with joblib bug: https://github.com/joblib/joblib/issues/1017"""
    import logging

    if len(logging.root.handlers) == 0:
        logging.basicConfig(
            filename="recalculate_metrics.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
        )
        multiprocessing_logging.install_mp_handler()


def path_is_empty(path: Path) -> bool:
    return not any(path.iterdir())


class MetricComputor:

    def __init__(self, results_path: Path,
                 data_paths: List[Path],
                 metrics: List[Metric],
                 save_to_dir: bool = False,
                 n_jobs: int = 1):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._n_jobs = n_jobs
        self.results_path = results_path.resolve()
        self.data_paths = data_paths
        self.metrics = metrics
        self.algos = self._build_algorithm_dict()
        self.dmgr = MultiDatasetManager(data_paths)
        self.df: pd.DataFrame = pd.read_csv(results_path / RESULTS_CSV)

        self._save_to_dir = save_to_dir
        if save_to_dir:
            self._logger.warning(f"The MetricComputor changes the results folder ({self.results_path}) in-place! "
                                 "If you do not want this, cancel this script using Ctrl-C! "
                                 f"Waiting {INITIAL_WAITING_SECONDS} seconds before continuing ...")
            time.sleep(INITIAL_WAITING_SECONDS)

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

    def recompute(self, recompute_existing: bool = False):
        exp_indices = self.df.index.values
        self._logger.info(f"Re-computing the metrics of {len(exp_indices)} experiments from folder {self.results_path}")
        with tqdm_joblib(tqdm(desc="Re-computing metrics", total=len(exp_indices))):
            updated_entries: List[pd.Series] = Parallel(n_jobs=self._n_jobs)(
                delayed(self._process_entry)(self.df.iloc[i].copy(), i, recompute_existing) for i in exp_indices
            )
        self.df = pd.DataFrame(updated_entries)
        self._logger.info(f"Overwriting results file at {self.results_path / RESULTS_CSV}")
        self.df.to_csv(self.results_path / RESULTS_CSV, index=False)

    def _process_entry(self, s_exp: pd.Series, i: int, recompute_existing: bool = False) -> pd.Series:
        init_logging()
        logger = logging.getLogger(f"{MetricComputor.__name__}.{multiprocessing.current_process().pid}")
        if s_exp.status in ["Status.ERROR", "Status.TIMEOUT"]:
            logger.info(f"Exp-{i:06d}: Skipping because experiment was not successful.")
            return s_exp

        exp_path = self._exp_path(s_exp)
        processed_scores_path = exp_path / ANOMALY_SCORES_TS
        metrics_path = exp_path / METRICS_CSV

        if not processed_scores_path.exists():
            logger.error(f"Exp-{i:06d}: Skipping because no anomaly scores found!")
            return s_exp

        logger.info(f"Exp-{i:06d}: Starting processing ...")
        y_true = load_labels_only(self.dmgr.get_dataset_path((s_exp.collection, s_exp.dataset)))
        y_scores = np.genfromtxt(processed_scores_path, delimiter=",")

        if not metrics_path.exists():
            metric_scores = {}
        else:
            metric_scores = pd.read_csv(metrics_path).iloc[0, :].to_dict()

        if recompute_existing:
            metric_list = [m.name for m in self.metrics]
            metrics_to_delete = [n for n in metric_scores if n not in metric_list and not n.endswith("time")]
            if len(metrics_to_delete) > 0:
                logger.warning(f"Exp-{i:06d}: Removing {', '.join(metrics_to_delete)}!")
                for m in metrics_to_delete:
                    del metric_scores[m]
        else:
            metric_list = [m.name for m in self.metrics if m.name not in metric_scores]
            if len(metric_list) == 0:
                logger.info(f"Exp-{i:06d}: ... skipping re-assessment of metrics, they are all present.")
                return s_exp

        results = {}
        errors = 0
        for metric in self.metrics:
            if metric.name not in metric_list:
                continue
            try:
                score = metric(y_true, y_scores)
                results[metric.name] = score
            except Exception as e:
                logger.warning(f"Exp-{i:06d}: Exception while computing metric {metric.name}!", exc_info=e)
                errors += 1
                continue

        # update metrics and write them to disk
        metric_scores.update(results)
        if metric_scores and self._save_to_dir:
            logger.debug(f"Exp-{i:06d}: Writing updated metrics to {metrics_path}!")
            pd.DataFrame([metric_scores]).to_csv(metrics_path, index=False)

        if metric_scores and errors == 0:
            logger.debug(f"Exp-{i:06d}: Updating metrics in index file.")
            for metric_name in metric_scores:
                s_exp[metric_name] = metric_scores[metric_name]
        else:
            logger.warning(f"Exp-{i:06d}: No metrics computed!")
        logger.info(f"Exp-{i:06d}: ... finished processing.")
        return s_exp

    def _exp_path(self, exp: pd.Series) -> Path:
        return (self.results_path
                / exp.algorithm
                / exp.hyper_params_id
                / exp.collection
                / exp.dataset
                / str(exp.repetition))


_metrics = {
    DefaultMetrics.ROC_AUC.name: DefaultMetrics.ROC_AUC,
    DefaultMetrics.PR_AUC.name: DefaultMetrics.PR_AUC,
    DefaultMetrics.RANGE_PR_AUC.name: DefaultMetrics.RANGE_PR_AUC,
    DefaultMetrics.FIXED_RANGE_PR_AUC.name: DefaultMetrics.FIXED_RANGE_PR_AUC,
    DefaultMetrics.AVERAGE_PRECISION.name: DefaultMetrics.AVERAGE_PRECISION,
    PrecisionAtK().name: PrecisionAtK(),
    FScoreAtK().name: FScoreAtK()
}


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Takes an experiment result folder and (re-)calculates the quality metrics."
    )
    parser.add_argument("result_folder", type=Path,
                        help="Folder of the experiment")
    parser.add_argument("data_folders", type=Path, nargs="*",
                        help="Folders, where the datasets from the experiment are stored.")
    parser.add_argument("--loglevel", default="INFO", choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        help="Set logging verbosity (default: %(default)s)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Set this flag if existing metrics should be recomputed as well. With this option, the "
                             "script will remove metrics not present in the metric list anymore!")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save the metrics in the experiments' result folders in addition to the results.csv")
    parser.add_argument("--metrics", type=str, nargs="*", default=["ROC_AUC", "PR_AUC", "RANGE_PR_AUC"],
                        choices=list(_metrics.keys()),
                        help="Metrics to re-calculate. (default: %(default)s)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Set the parallelism. -1 uses all available cores.")
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = _create_arg_parser()
    logging.basicConfig(level=args.loglevel)
    selected_metrics = args.metrics
    selected_metrics = [_metrics[m] for m in selected_metrics]

    rs = MetricComputor(args.result_folder, args.data_folders, selected_metrics, args.save, args.n_jobs)
    rs.recompute(args.force)
