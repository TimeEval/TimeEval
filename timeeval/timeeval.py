import asyncio
import logging
from pathlib import Path
from typing import Callable
from enum import Enum
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tqdm
from io import StringIO
import sys
from contextlib import redirect_stdout
import datetime as dt
from distributed.client import Future
import subprocess
import socket

from .adapters import DockerAdapter
from .adapters.base import BaseAdapter
from timeeval.datasets import Datasets
from timeeval.utils.metrics import roc
from .algorithm import Algorithm
from .data_types import AlgorithmParameter
from .remote import Remote
from .times import Times

METRICS_CSV = "metrics.csv"
EXECUTION_LOG = "execution.log"
ANOMALY_SCORES_TS = "anomaly_scores.ts"


class Status(Enum):
    OK = 0
    ERROR = 1
    TIMEOUT = 2  # not yet implemented


class TimeEval:
    RESULT_KEYS = ("algorithm",
                   "collection",
                   "dataset",
                   "score",
                   "preprocess_time",
                   "main_time",
                   "postprocess_time",
                   "status",
                   "error_message",
                   "repetition")

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 results_path: Path = Path("./results"),
                 distributed: bool = False,
                 ssh_cluster_kwargs: Optional[dict] = None,
                 repetitions: int = 1):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.dmgr = dataset_mgr
        self.results_path = results_path.absolute()
        self.start_date: Optional[str] = None

        self.distributed = distributed
        self.cluster_kwargs = ssh_cluster_kwargs or {}
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS)
        self.repetitions = repetitions

        if self.distributed:
            self.remote = Remote(**self.cluster_kwargs)
            self.results["future_result"] = np.nan

    def _gen_args(self, algorithm_name: str, dataset_name: Tuple[str, str], repetition: int) -> dict:
        assert self.start_date, "The start date isn't set! Run TimeEval.run() first!"
        results_path = self.results_path / self.start_date / algorithm_name / dataset_name[0] / dataset_name[1] / str(repetition)

        return {
            "results_path": results_path
        }

    def _load_dataset(self, name: Tuple[str, str]) -> pd.DataFrame:
        return self.dmgr.get_dataset_df(name)

    def _get_dataset_path(self, name: Tuple[str, str]) -> Path:
        return self.dmgr.get_dataset_path(name, train=False)

    def _get_X_and_y(self, dataset_name: Tuple[str, str], data_as_file: bool = False) -> Tuple[AlgorithmParameter, np.ndarray]:
        dataset = self._load_dataset(dataset_name)
        if data_as_file:
            X = self._get_dataset_path(dataset_name)
        else:
            if dataset.shape[1] > 3:
                X = dataset.values[:, 1:-1]
            elif dataset.shape[1] == 3:
                X = dataset.values[:, 1]
            else:
                raise ValueError(f"Dataset '{dataset_name}' has a shape that was not expected: {dataset.shape}")
        y = dataset.values[:, -1]
        return X, y

    def _run_algorithm(self, algorithm: Algorithm):
        if not self.distributed:
            pbar = tqdm.tqdm(self.dataset_names, desc=f"Evaluating {algorithm.name}", position=1)
        for dataset_name in self.dataset_names:
            for repetition in range(1, self.repetitions + 1):
                try:
                    future_result: Optional[Future] = None
                    result: Optional[Dict] = None

                    X, y_true = self._get_X_and_y(dataset_name, data_as_file=algorithm.data_as_file)
                    args = self._gen_args(algorithm.name, dataset_name, repetition)

                    if self.distributed:
                        future_result = self.remote.add_task(TimeEval.evaluate, algorithm, X, y_true, args)
                    else:
                        result = TimeEval.evaluate(algorithm, X, y_true, args)
                    self._record_results(algorithm.name, dataset_name, result, future_result, repetition=repetition)

                except Exception as e:
                    logging.exception(
                        f"Exception occured during the evaluation of {algorithm.name} on the dataset {dataset_name}:")
                    f: asyncio.Future = asyncio.Future()
                    f.set_result({
                        "score": np.nan,
                        "main_time": np.nan,
                        "preprocess_time": np.nan,
                        "postprocess_time": np.nan
                    })
                    self._record_results(algorithm.name, dataset_name,
                                         future_result=f,
                                         status=Status.ERROR,
                                         error_message=str(e))

            if not self.distributed:
                pbar.update()
        if not self.distributed:
            pbar.close()

    @staticmethod
    def evaluate(algorithm: Algorithm, X: AlgorithmParameter, y_true: np.ndarray, args: dict) -> Dict:
        results_path = args.get("results_path", Path("./results"))
        results_path.mkdir(parents=True, exist_ok=True)

        logs_file = (results_path / EXECUTION_LOG).open("w")
        with redirect_stdout(logs_file):
            y_scores, times = Times.from_algorithm(algorithm, X, args)
        score = roc(y_scores, y_true.astype(np.float64), plot=False)
        result = {"score": score}
        result.update(times.to_dict())

        y_scores.tofile(results_path / ANOMALY_SCORES_TS, sep="\n")
        pd.DataFrame([result]).to_csv(results_path / METRICS_CSV, index=False)

        return result

    def _record_results(self,
                        algorithm_name: str,
                        dataset_name: Tuple[str, str],
                        result: Optional[Dict] = None,
                        future_result: Optional[Future] = None,
                        status: Status = Status.OK,
                        error_message: Optional[str] = None,
                        repetition: int = 1):
        new_row = {
            "algorithm": algorithm_name,
            "collection": dataset_name[0],
            "dataset": dataset_name[1],
            "status": status.name,
            "error_message": error_message,
            "repetition": repetition
        }
        if result is not None and future_result is None:
            new_row.update(result)
        elif result is None and future_result is not None:
            new_row.update({"future_result": future_result})
        self.results = self.results.append(new_row, ignore_index=True)
        self.results.replace(to_replace=[None], value=np.nan, inplace=True)

    def _get_future_results(self):
        self.remote.fetch_results()

        keys = ["score", "preprocess_time", "main_time", "postprocess_time"]

        def get_future_result(f: Future) -> List[float]:
            r = f.result()
            return [r[k] for k in keys]

        self.results[keys] = self.results["future_result"].apply(get_future_result).tolist()
        self.results = self.results.drop(['future_result'], axis=1)

    def get_results(self, aggregated: bool = True) -> pd.DataFrame:
        if not aggregated:
            return self.results

        df = self.results

        if Status.ERROR.name in df.status.unique():
            logging.warning("The results contain errors which are filtered out for the final aggregation. "
                            "To see all results, call .get_results(aggregated=False)")
            df = df[df.status == Status.OK.name]

        keys = ["score", "preprocess_time", "main_time", "postprocess_time"]
        grouped_results = df.groupby(["algorithm", "collection", "dataset"])
        repetitions = [len(v) for k, v in grouped_results.groups.items()]
        mean_results: pd.DataFrame = grouped_results.mean()[keys]
        std_results = grouped_results.std()[keys]
        results = mean_results.join(std_results, lsuffix="_mean", rsuffix="_std")
        results["repetitions"] = repetitions
        return results

    def save_results(self, results_path: Optional[Path] = None):
        results_path = results_path or (self.results_path / Path("results.csv"))
        self.results.to_csv(results_path, index=False)

    def rsync_results(self):
        hosts = self.cluster_kwargs.get("hosts", list())
        hostname = socket.gethostname()
        for host in hosts:
            if host != hostname:
                subprocess.call(["rsync", "-a", f"{self.results_path}/", f"{host}:{self.results_path}"])

    def _prune_docker(self):
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.algorithms:
            if isinstance(algorithm.main, DockerAdapter):
                tasks.append((algorithm.main.prune, [], {}))
        self.remote.run_on_all_hosts(tasks)

    def _distributed_prepare(self):
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.algorithms:
            if isinstance(algorithm.main, BaseAdapter):
                tasks.append((algorithm.main.prepare, [], {}))
            for dataset_name in self.dataset_names:
                for repetition in range(self.repetitions):
                    tasks.append((self._gen_args(algorithm.name, dataset_name, repetition)
                                  .get("results_path", Path("./results")).mkdir,
                                  [], {"parents": True, "exist_ok": True}))
        self.remote.run_on_all_hosts(tasks)

    def _distributed_execute(self):
        for algorithm in self.algorithms:
            self._run_algorithm(algorithm)

    def _distributed_finalize(self):
        self._get_future_results()
        self._prune_docker()
        self.remote.close()
        self.rsync_results()

    def run(self):
        assert len(self.algorithms) > 0, "No algorithms given for evaluation"
        self.start_date = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if self.distributed:
            self._distributed_prepare()
            self._distributed_execute()
            self._distributed_finalize()
        else:
            for algorithm in tqdm.tqdm(self.algorithms, desc="Evaluating Algorithms", position=0):
                self._run_algorithm(algorithm)
