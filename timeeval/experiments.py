import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Generator, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .algorithm import Algorithm
from .constants import EXECUTION_LOG, ANOMALY_SCORES_TS, METRICS_CSV, HYPER_PARAMETERS
from .data_types import AlgorithmParameter
from .resource_constraints import ResourceConstraints
from .times import Times
from .utils.datasets import extract_features, extract_labels, load_dataset
from .utils.hash_dict import hash_dict
from .utils.metrics import Metric


@dataclass
class Experiment:
    dataset: Tuple[str, str]
    algorithm: Algorithm
    params: dict
    repetition: int
    base_results_dir: Path
    resource_constraints: ResourceConstraints
    metrics: List[Metric]

    @property
    def dataset_collection(self) -> str:
        return self.dataset[0]

    @property
    def dataset_name(self) -> str:
        return self.dataset[1]

    @property
    def params_id(self) -> str:
        return hash_dict(self.params)

    @property
    def results_path(self) -> Path:
        return (self.base_results_dir / self.algorithm.name / self.params_id / self.dataset_collection /
                self.dataset_name / str(self.repetition))

    def build_args(self) -> dict:
        return {
            "results_path": self.results_path,
            "resource_constraints": self.resource_constraints,
            "hyper_params": self.params
        }

    def evaluate(self, resolved_dataset_path: Path) -> dict:
        dataset = load_dataset(resolved_dataset_path)
        y_true = extract_labels(dataset)
        if self.algorithm.data_as_file:
            X: AlgorithmParameter = resolved_dataset_path
        else:
            if dataset.shape[1] >= 3:
                X = extract_features(dataset)
            else:
                raise ValueError(
                    f"Dataset '{resolved_dataset_path.name}' has a shape that was not expected: {dataset.shape}")

        with (self.results_path / EXECUTION_LOG).open("w") as logs_file, redirect_stdout(logs_file):
            y_scores, times = Times.from_algorithm(self.algorithm, X, self.build_args())
        # scale scores to [0, 1]
        if len(y_scores.shape) == 1:
            y_scores = y_scores.reshape(-1, 1)
        y_scores = MinMaxScaler().fit_transform(y_scores).reshape(-1)
        result = {}
        for metric in self.metrics:
            score = metric(y_scores, y_true.astype(np.float64))
            result[metric.name] = score
        result.update(times.to_dict())

        y_scores.tofile(str(self.results_path / ANOMALY_SCORES_TS), sep="\n")
        pd.DataFrame([result]).to_csv(self.results_path / METRICS_CSV, index=False)

        with (self.results_path / HYPER_PARAMETERS).open("w") as f:
            json.dump(self.params, f)

        return result


class Experiments:

    def __init__(self, datasets: List[Tuple[str, str]], algorithms: List[Algorithm], base_result_path: Path,
                 resource_constraints: ResourceConstraints, repetitions: int = 1, metrics: Optional[List[Metric]] = None):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions
        self.base_result_path = base_result_path
        self.resource_constraints = resource_constraints
        self.metrics = metrics or Metric.DEFAULT_METRICS

    def __iter__(self) -> Generator[Experiment, None, None]:
        for algorithm in self.algorithms:
            for algorithm_config in algorithm.param_grid:
                for dataset_name in self.dataset_names:
                    for repetition in range(1, self.repetitions + 1):
                        yield Experiment(
                            algorithm=algorithm,
                            dataset=dataset_name,
                            params=algorithm_config,
                            repetition=repetition,
                            base_results_dir=self.base_result_path,
                            resource_constraints=self.resource_constraints,
                            metrics=self.metrics
                        )

    def __len__(self) -> int:
        return sum([len(algo.param_grid) for algo in self.algorithms]) * len(self.dataset_names) * self.repetitions
