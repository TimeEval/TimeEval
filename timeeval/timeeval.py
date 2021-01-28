import time
import logging
from pathlib import Path
from typing import List, Callable, Tuple, Any, Dict, Union, Optional
from distributed.client import Future
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import tqdm

from .remote import Remote
from timeeval.datasets import Datasets
from timeeval.utils.metrics import roc

AlgorithmParameter = Union[np.ndarray, Path]
TSFunction = Callable[[AlgorithmParameter, dict], AlgorithmParameter]
TSFunctionPost = Callable[[AlgorithmParameter, dict], np.ndarray]


@dataclass
class Algorithm:
    name: str
    main: TSFunction
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False


@dataclass
class Times:
    main: float
    preprocess: Optional[float] = None
    postprocess: Optional[float] = None

    def to_dict(self) -> Dict:
        return {f"{k}_time": v for k, v in asdict(self).items()}

    @staticmethod
    def from_algorithm(algorithm: Algorithm, X: AlgorithmParameter, args: dict) -> Tuple[np.ndarray, 'Times']:
        x, pre_time = timer(algorithm.preprocess, X, args) if algorithm.preprocess else (X, None)
        x, main_time = timer(algorithm.main, x, args)
        x, post_time = timer(algorithm.postprocess, x, args) if algorithm.postprocess else(x, None)
        return x, Times(main_time, preprocess=pre_time, postprocess=post_time)


def timer(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    fn_result = fn(*args, **kwargs)
    end = time.time()
    duration = end - start
    return fn_result, duration


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
                   "error_message")

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 results_path: Path = Path("./results"),
                 distributed: bool = False,
                 ssh_cluster_kwargs: Optional[dict] = None):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.dmgr = dataset_mgr
        self.results_path = results_path

        self.distributed = distributed
        self.cluster_kwargs = ssh_cluster_kwargs or {}
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS)

        if self.distributed:
            self.remote = Remote(**self.cluster_kwargs)
            self.results["future_result"] = np.nan

    def _gen_args(self, algorithm_name: str, dataset_name: Tuple[str, str]) -> dict:
        return {
            "results_path": self.results_path / algorithm_name / dataset_name[0] / dataset_name[1]
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
        for dataset_name in tqdm.tqdm(self.dataset_names, desc=f"Evaluating {algorithm.name}", position=1):
            try:
                future_result: Optional[Future] = None
                result: Optional[Dict] = None

                X, y_true = self._get_X_and_y(dataset_name, data_as_file=algorithm.data_as_file)
                args = self._gen_args(algorithm.name, dataset_name)

                if self.distributed:
                    future_result = self.remote.add_task(TimeEval.evaluate, algorithm, X, y_true, args)
                else:
                    result = TimeEval.evaluate(algorithm, X, y_true, args)
                self._record_results(algorithm.name, dataset_name, result, future_result)

            except Exception as e:
                logging.error(
                    f"Exception occured during the evaluation of {algorithm.name} on the dataset {dataset_name}:")
                logging.error(str(e))
                self._record_results(algorithm.name, dataset_name, status=Status.ERROR, error_message=str(e))

    @staticmethod
    def evaluate(algorithm: Algorithm, X: AlgorithmParameter, y_true: np.ndarray, args: dict) -> Dict:
        y_scores, times = Times.from_algorithm(algorithm, X, args)
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

    def save_results(self, results_path: Optional[Path] = None):
        results_path = results_path or (self.results_path / Path("results.csv"))
        self.results.to_csv(results_path, index=False)

    def run(self):
        assert len(self.algorithms) > 0, "No algorithms given for evaluation"

        for algorithm in tqdm.tqdm(self.algorithms, desc="Evaluating Algorithms", position=0):
            self._run_algorithm(algorithm)

        if self.distributed:
            self._get_future_results()
            self.remote.close()
