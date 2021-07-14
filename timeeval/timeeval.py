import asyncio
import datetime as dt
import json
import logging
import signal
import socket
import subprocess
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import tqdm
from distributed.client import Future
from joblib import Parallel, delayed

from .adapters.docker import DockerTimeoutError
from .algorithm import Algorithm
from .constants import RESULTS_CSV
from .data_types import TrainingType
from .datasets import Datasets
from .experiments import Experiments, Experiment
from .remote import Remote, RemoteConfiguration
from .resource_constraints import ResourceConstraints
from .times import Times
from .utils.encode_params import dumps_params
from .utils.metrics import Metric
from .utils.tqdm_joblib import tqdm_joblib


class Status(Enum):
    OK = 0
    ERROR = 1
    TIMEOUT = 2


class TimeEval:
    RESULT_KEYS = ["algorithm",
                   "collection",
                   "dataset",
                   "algo_training_type",
                   "algo_input_dimensionality",
                   "dataset_training_type",
                   "dataset_input_dimensionality",
                   "train_preprocess_time",
                   "train_main_time",
                   "execute_preprocess_time",
                   "execute_main_time",
                   "execute_postprocess_time",
                   "status",
                   "error_message",
                   "repetition",
                   "hyper_params",
                   "hyper_params_id"]

    DEFAULT_RESULT_PATH = Path("./results")

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 results_path: Path = DEFAULT_RESULT_PATH,
                 repetitions: int = 1,
                 distributed: bool = False,
                 remote_config: Optional[RemoteConfiguration] = None,
                 resource_constraints: Optional[ResourceConstraints] = None,
                 disable_progress_bar: bool = False,
                 metrics: Optional[List[Metric]] = None,
                 skip_invalid_combinations: bool = True,
                 force_training_type_match: bool = False,
                 n_jobs: int = -1):
        self.log = logging.getLogger(self.__class__.__name__)
        start_date: str = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        resource_constraints = resource_constraints or ResourceConstraints()
        if not distributed and resource_constraints.tasks_per_host > 1:
            self.log.warning(
                f"`tasks_per_host` was set to {resource_constraints.tasks_per_host}. However, non-distributed "
                "execution of TimeEval does currently not support parallelism! Reducing `tasks_per_host` to 1. "
                "The automatic resource limitation will reflect this by increasing the limits for the single task. "
                "Explicitly set constraints to limit the resources for local executions of TimeEval."
            )
            resource_constraints.tasks_per_host = 1
        self.disable_progress_bar = disable_progress_bar

        self.results_path = results_path.absolute() / start_date
        self.log.info(f"Results are recorded in the directory {self.results_path}")
        self.metrics = metrics or Metric.default_list()
        self.metric_names = [m.name for m in self.metrics]
        dataset_details = [dataset_mgr.get(d) for d in datasets]
        self.exps = Experiments(dataset_mgr, dataset_details, algorithms, self.results_path,
                                resource_constraints=resource_constraints,
                                repetitions=repetitions,
                                skip_invalid_combinations=skip_invalid_combinations,
                                force_training_type_match=force_training_type_match,
                                metrics=self.metrics)
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS + self.metric_names)

        self.distributed = distributed
        self.remote_config = remote_config or RemoteConfiguration()
        self.n_jobs = n_jobs

        if self.distributed:
            self.log.info("TimeEval is running in distributed environment, setting up remoting ...")
            self.remote = Remote(disable_progress_bar=self.disable_progress_bar, remote_config=self.remote_config,
                                 resource_constraints=resource_constraints)
            self.results["future_result"] = np.nan

            self.log.info("... registering signal handlers ...")
            orig_handler: Callable[[signal.Signals, FrameType], Any] = signal.getsignal(signal.SIGINT)  # type: ignore

            def sigint_handler(sig: signal.Signals, frame: FrameType):
                self.log.warning(f"SIGINT ({sig}) received, shutting down cluster. Please look for dangling Docker "
                                 "containers on all worker nodes (we do not remove them when terminating "
                                 "ungracefully).")
                self.remote.close()
                return orig_handler(sig, frame)
            signal.signal(signal.SIGINT, sigint_handler)

            self.log.info("... remoting setup done.")

    def _run(self):
        desc = "Submitting evaluation tasks" if self.distributed else "Evaluating"
        for exp in tqdm.tqdm(self.exps, desc=desc, disable=self.disable_progress_bar):
            try:
                future_result: Optional[Future] = None
                result: Optional[Dict] = None

                if exp.algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]:
                    if exp.resolved_train_dataset_path is None:
                        # Intentionally raise KeyError here if no training dataset is specified.
                        # The Error will be caught by the except clause below.
                        raise KeyError("Path to training dataset not found!")

                    # This check is not necessary for unsupervised algorithms, because they can be executed on all
                    # datasets.
                    if exp.algorithm.training_type != exp.dataset.training_type:
                        raise ValueError(f"Dataset training type ({exp.dataset.training_type}) incompatible to "
                                         f"algorithm training type ({exp.algorithm.training_type})!")

                if exp.algorithm.input_dimensionality != exp.dataset.input_dimensionality:
                    raise ValueError(f"Dataset input dimensionality ({exp.dataset.input_dimensionality}) incompatible "
                                     f"to algorithm input dimensionality ({exp.algorithm.input_dimensionality})!")

                if self.distributed:
                    future_result = self.remote.add_task(exp.evaluate, key=exp.name)
                else:
                    result = exp.evaluate()
                self._record_results(exp, result=result, future_result=future_result)

            except DockerTimeoutError as e:
                self.log.exception(
                    f"Evaluation of {exp.algorithm.name} on the dataset {exp.dataset} timed out.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.TIMEOUT, error_message=str(e))
            except Exception as e:
                self.log.exception(
                    f"Exception occurred during the evaluation of {exp.algorithm.name} on the dataset {exp.dataset}.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.ERROR, error_message=str(e))

    def _record_results(self,
                        exp: Experiment,
                        result: Optional[Dict] = None,
                        future_result: Optional[Future] = None,
                        status: Status = Status.OK,
                        error_message: Optional[str] = None):
        new_row = {
            "algorithm": exp.algorithm.name,
            "collection": exp.dataset_collection,
            "dataset": exp.dataset_name,
            "algo_training_type": exp.algorithm.training_type.name,
            "algo_input_dimensionality": exp.algorithm.input_dimensionality.name,
            "dataset_training_type": exp.dataset.training_type.name,
            "dataset_input_dimensionality": exp.dataset.input_dimensionality.name,
            "status": status.name,
            "error_message": error_message,
            "repetition": exp.repetition,
            "hyper_params": dumps_params(exp.params),
            "hyper_params_id": exp.params_id
        }
        if result is not None and future_result is None:
            new_row.update(result)
        elif result is None and future_result is not None:
            new_row.update({"future_result": future_result})
        self.results = self.results.append(new_row, ignore_index=True)
        self.results.replace(to_replace=[None], value=np.nan, inplace=True)

    def _resolve_future_results(self):
        self.remote.fetch_results()

        result_keys = self.metric_names + Times.result_keys()
        status_keys = ["status", "error_message"]
        keys = result_keys + status_keys

        def get_future_result(f: Future) -> Tuple:
            try:
                r = f.result()
                return tuple(r.get(k, None) for k in result_keys) + (Status.OK, None)
            except DockerTimeoutError as e:
                self.log.exception(f"Exception {str(e)} occurred remotely.")
                status = Status.TIMEOUT
                error_message = str(e)
            except Exception as e:
                self.log.exception(f"Exception {str(e)} occurred remotely.")
                status = Status.ERROR
                error_message = str(e)

            return tuple(np.nan for _ in result_keys) + (status, error_message)

        self.results[keys] = self.results["future_result"].apply(get_future_result).tolist()
        self.results = self.results.drop(['future_result'], axis=1)

    def get_results(self, aggregated: bool = True, short: bool = True) -> pd.DataFrame:
        if not aggregated:
            return self.results

        df = self.results

        if Status.ERROR.name in df.status.unique() or Status.TIMEOUT.name in df.status.unique():
            self.log.warning("The results contain errors which are filtered out for the final aggregation. "
                             "To see all results, call .get_results(aggregated=False)")
            df = df[df.status == Status.OK.name]

        if short:
            time_names = ["train_main_time", "execute_main_time"]
            group_names = ["algorithm", "collection", "dataset"]
        else:
            time_names = [key for key in Times.result_keys() if key in df.columns]
            group_names = ["algorithm", "collection", "dataset", "hyper_params_id"]
        keys = self.metric_names + time_names
        grouped_results = df.groupby(group_names)
        repetitions = [len(v) for k, v in grouped_results.groups.items()]
        results: pd.DataFrame = grouped_results.mean()[keys]

        if short:
            results = results.rename(columns=dict([(k, f"{k}_mean") for k in keys]))
        else:
            std_results = grouped_results.std()[keys]
            results = results.join(std_results, lsuffix="_mean", rsuffix="_std")
        results["repetitions"] = repetitions
        return results

    def save_results(self, results_path: Optional[Path] = None):
        path = results_path or (self.results_path / RESULTS_CSV)
        self.results.to_csv(path.absolute(), index=False)

    @staticmethod
    def rsync_results(results_path: Path, hosts: List[str], disable_progress_bar: bool = False, n_jobs: int = -1):
        excluded_aliases = [
            hostname := socket.gethostname(),
            socket.gethostbyname(hostname),
            "localhost",
            socket.gethostbyname("localhost")
        ]
        jobs = [
            delayed(subprocess.call)(["rsync", "-a", f"{host}:{results_path}/", f"{results_path}"])
            for host in hosts
            if host not in excluded_aliases
        ]
        with tqdm_joblib(tqdm.tqdm(hosts, desc="Collecting results", disable=disable_progress_bar, total=len(jobs))):
            Parallel(n_jobs)(jobs)

    def _rsync_results(self):
        TimeEval.rsync_results(
            self.results_path,
            self.remote_config.worker_hosts,
            self.disable_progress_bar,
            self.n_jobs
        )

    def _prepare(self):
        n = len(self.exps)
        self.log.debug(f"Running {n} algorithm prepare steps")
        for algorithm in self.exps.algorithms:
            algorithm.prepare()
        self.log.debug(f"Creating {n} result directories")
        for exp in self.exps:
            exp.results_path.mkdir(parents=True, exist_ok=True)

    def _finalize(self):
        self.log.debug(f"Running {len(self.exps)} algorithm finalize steps")
        for algorithm in self.exps.algorithms:
            algorithm.finalize()

    def _distributed_prepare(self):
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            if prepare_fn := algorithm.prepare_fn():
                tasks.append((prepare_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm prepare steps")

        def mkdirs(dirs: List[Path]) -> None:
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)

        dir_list = [exp.results_path for exp in self.exps]
        tasks.append((mkdirs, [dir_list], {}))
        self.log.debug(f"Collected {len(dir_list)} directories to create on remote nodes")
        self.remote.run_on_all_hosts(tasks, msg="Preparing")

    def _distributed_finalize(self):
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            if finalize_fn := algorithm.finalize_fn():
                tasks.append((finalize_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm finalize steps")
        self.remote.run_on_all_hosts(tasks, msg="Finalizing")
        self.remote.close()
        self._rsync_results()

    def run(self):
        assert len(self.exps.algorithms) > 0, "No algorithms given for evaluation"

        print("Running PREPARE phase")
        self.log.info("Running PREPARE phase")
        if self.distributed:
            self._distributed_prepare()
        else:
            self._prepare()

        print("Running EVALUATION phase")
        self.log.info("Running EVALUATION phase")
        self._run()
        if self.distributed:
            self._resolve_future_results()
        self.save_results()

        print("Running FINALIZE phase")
        self.log.info("Running FINALIZE phase")
        if self.distributed:
            self._distributed_finalize()
        else:
            self._finalize()

        msg = f"FINALIZE phase done. Stored results at {self.results_path / RESULTS_CSV}"
        print(msg)
        self.log.info(msg)
