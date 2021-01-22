import logging
import time
from pathlib import Path
from typing import List, Callable, Tuple, Any, NamedTuple, Dict, Union, Optional
from distributed.client import Future
from enum import Enum
import numpy as np
import pandas as pd
import tqdm

from .remote import Remote
from timeeval.datasets import Datasets
from timeeval.utils.metrics import roc


class Algorithm(NamedTuple):
    name: str
    function: Callable[[Union[np.ndarray, Path]], np.ndarray]
    data_as_file: bool


class Times(NamedTuple):
    preprocess: Optional[float]
    main: float
    postprocess: Optional[float]

    def to_dict(self) -> Dict:
        return {f"{k}_time": v for k, v in dict(self._asdict()).items()}


class Status(Enum):
    OK = 0
    ERROR = 1
    TIMEOUT = 2  # not yet implemented


def timer(fn: Callable, *args, **kwargs) -> Tuple[Any, Times]:
    start = time.time()
    fn_result = fn(*args, **kwargs)
    end = time.time()
    duration = end - start
    return fn_result, Times(preprocess=None, main=duration, postprocess=None)


class TimeEval:
    RESULT_KEYS = ("algorithm",
                   "collection",
                   "dataset",
                   "score",
                   "preprocess_time",
                   "main_time",
                   "postprocess_time",
                   "status",
                   "error_message")

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 distributed: bool = False,
                 ssh_cluster_kwargs: Optional[dict] = None):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.dmgr = dataset_mgr

        self.distributed = distributed
        self.cluster_kwargs = ssh_cluster_kwargs or {}
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS)

        if self.distributed:
            self.remote = Remote(**self.cluster_kwargs)
            self.results["future_result"] = np.nan

    def _load_dataset(self, name: Tuple[str, str]) -> pd.DataFrame:
        return self.dmgr.get_dataset_df(name)

    def _get_dataset_path(self, name: Tuple[str, str]) -> Path:
        return self.dmgr.get_dataset_path(name, train=False)

    def _run_algorithm(self, algorithm: Algorithm):
        for dataset_name in tqdm.tqdm(self.dataset_names, desc=f"Evaluating {algorithm.name}", position=1):
            if algorithm.data_as_file:
                dataset_file = self._get_dataset_path(dataset_name)
                self._run_from_data_file(algorithm, dataset_file, dataset_name)
            else:
                dataset = self._load_dataset(dataset_name)
                self._run_w_loaded_data(algorithm, dataset, dataset_name)

    def _run_from_data_file(self, algorithm: Algorithm, dataset_file: Path, dataset_name: str):
        raise NotImplementedError()

    def _run_w_loaded_data(self, algorithm: Algorithm, dataset: pd.DataFrame, dataset_name: Tuple[str, str]):
        try:
            future_result: Optional[Future] = None
            result: Optional[Dict] = None

            if self.distributed:
                future_result = self.remote.add_task(TimeEval.evaluate, algorithm, dataset, dataset_name)
            else:
                result = TimeEval.evaluate(algorithm, dataset, dataset_name)
            self._record_results(algorithm.name, dataset_name, result, future_result)

        except Exception as e:
            logging.error(f"Exception occured during the evaluation of {algorithm.name} on the dataset {dataset_name}:")
            logging.error(str(e))
            self._record_results(algorithm.name, dataset_name, status=Status.ERROR, error_message=str(e))

    @staticmethod
    def evaluate(algorithm: Algorithm, dataset: pd.DataFrame, dataset_name: Tuple[str, str]) -> Dict:
        y_true = dataset.values[:, -1]
        if dataset.shape[1] > 3:
            X = dataset.values[:, 1:-1]
        elif dataset.shape[1] == 3:
            X = dataset.values[:, 1]
        else:
            raise ValueError(f"Dataset '{dataset_name}' has a shape that was not expected: {dataset.shape}")
        y_scores, times = timer(algorithm.function, X)
        score = roc(y_scores, y_true.astype(np.float), plot=False)
        result = {"score": score}
        result.update(times.to_dict())
        return result

    def _record_results(self,
                        algorithm_name: str,
                        dataset_name: Tuple[str, str],
                        result: Optional[Dict] = None,
                        future_result: Optional[Future] = None,
                        status: Status = Status.OK,
                        error_message: Optional[str] = None):
        new_row = {
            "algorithm": algorithm_name,
            "collection": dataset_name[0],
            "dataset": dataset_name[1],
            "status": status.name,
            "error_message": error_message
        }
        if result is not None and future_result is None:
            new_row.update(result)
        elif result is None and future_result is not None:
            new_row.update({"future_result": future_result})
        self.results = self.results.append(new_row, ignore_index=True)
        self.results.replace(to_replace=[None], value=np.nan, inplace=True)

    def _get_future_results(self):
        keys = ["score", "preprocess_time", "main_time", "postprocess_time"]

        def get_future_result(f: Future) -> List[float]:
            r = f.result()
            return [r[k] for k in keys]

        self.remote.fetch_results()
        self.results[keys] = self.results["future_result"].apply(get_future_result).tolist()
        self.results = self.results.drop(['future_result'], axis=1)

    def run(self):
        assert len(self.algorithms) > 0, "No algorithms given for evaluation"

        for algorithm in tqdm.tqdm(self.algorithms, desc="Evaluating Algorithms", position=0):
            self._run_algorithm(algorithm)

        if self.distributed:
            self._get_future_results()
            self.remote.close()
