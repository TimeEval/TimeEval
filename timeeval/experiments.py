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
from .data_types import AlgorithmParameter, TrainingType
from .datasets.datasets import Dataset
from .resource_constraints import ResourceConstraints
from .times import Times
from .utils.datasets import extract_features, extract_labels, load_dataset
from .utils.hash_dict import hash_dict
from .utils.metrics import Metric


@dataclass
class Experiment:
    dataset: Dataset
    algorithm: Algorithm
    params: dict
    repetition: int
    base_results_dir: Path
    resource_constraints: ResourceConstraints
    metrics: List[Metric]

    @property
    def dataset_collection(self) -> str:
        return self.dataset.collection_name

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

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

    def evaluate(self, resolved_train_dataset_path: Optional[Path], resolved_test_dataset_path: Path) -> dict:
        """
        Using TimeEval distributed, this method is executed on the remote node.
        """
        # perform training if necessary
        result = self._perform_training(resolved_train_dataset_path)

        # perform execution
        dataset = load_dataset(resolved_test_dataset_path)
        y_true = extract_labels(dataset)
        y_scores, execution_times = self._perform_execution(dataset, resolved_test_dataset_path)
        result.update(execution_times)

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file:
            print(f"Scoring algorithm {self.algorithm.name} with {','.join([m.name for m in self.metrics])} metrics",
                  file=logs_file)

        # scale scores to [0, 1]
        if len(y_scores.shape) == 1:
            y_scores = y_scores.reshape(-1, 1)
        y_scores = MinMaxScaler().fit_transform(y_scores).reshape(-1)

        # calculate quality metrics
        for metric in self.metrics:
            score = metric(y_scores, y_true.astype(np.float64))
            result[metric.name] = score

        y_scores.tofile(str(self.results_path / ANOMALY_SCORES_TS), sep="\n")
        pd.DataFrame([result]).to_csv(self.results_path / METRICS_CSV, index=False)

        with (self.results_path / HYPER_PARAMETERS).open("w") as f:
            json.dump(self.params, f)

        return result

    def _perform_training(self, train_dataset_path: Optional[Path]) -> dict:
        if self.algorithm.training_type == TrainingType.UNSUPERVISED:
            return {}

        if not train_dataset_path:
            raise ValueError(f"No training dataset was provided. Algorithm cannot be trained!")

        if self.algorithm.data_as_file:
            X: AlgorithmParameter = train_dataset_path
        else:
            X = load_dataset(train_dataset_path).values

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing training for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            times = Times.from_train_algorithm(self.algorithm, X, self.build_args())
        return times.to_dict()

    def _perform_execution(self, dataset: pd.DataFrame, dataset_path: Path) -> Tuple[np.ndarray, dict]:
        if self.algorithm.data_as_file:
            X: AlgorithmParameter = dataset_path
        else:
            if dataset.shape[1] >= 3:
                X = extract_features(dataset)
            else:
                raise ValueError(
                    f"Dataset '{dataset_path.name}' has a shape that was not expected: {dataset.shape}")

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing execution for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            y_scores, times = Times.from_execute_algorithm(self.algorithm, X, self.build_args())
        return y_scores, times.to_dict()


class Experiments:
    def __init__(self,
                 datasets: List[Dataset],
                 algorithms: List[Algorithm],
                 base_result_path: Path,
                 resource_constraints: ResourceConstraints,
                 repetitions: int = 1,
                 metrics: Optional[List[Metric]] = None,
                 skip_invalid_combinations: bool = False):
        self.datasets = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions
        self.base_result_path = base_result_path
        self.resource_constraints = resource_constraints
        self.metrics = metrics or Metric.default()
        self.skip_invalid_combinations = skip_invalid_combinations
        if self.skip_invalid_combinations:
            self._N: Optional[int] = None
        else:
            self._N: Optional[int] = sum(
                [len(algo.param_grid) for algo in self.algorithms]
            ) * len(self.datasets) * self.repetitions

    def __iter__(self) -> Generator[Experiment, None, None]:
        for algorithm in self.algorithms:
            for algorithm_config in algorithm.param_grid:
                for dataset in self.datasets:
                    for repetition in range(1, self.repetitions + 1):
                        if self._check_compatible(dataset, algorithm):
                            yield Experiment(
                                algorithm=algorithm,
                                dataset=dataset,
                                params=algorithm_config,
                                repetition=repetition,
                                base_results_dir=self.base_result_path,
                                resource_constraints=self.resource_constraints,
                                metrics=self.metrics
                            )

    def __len__(self) -> int:
        if not self._N:
            self._N = sum([
                1 for algorithm in self.algorithms
                for _algorithm_config in algorithm.param_grid
                for dataset in self.datasets
                for _repetition in range(1, self.repetitions + 1)
                if self._check_compatible(dataset, algorithm)
            ])
        return self._N

    def _check_compatible(self, dataset: Dataset, algorithm: Algorithm) -> bool:
        if not self.skip_invalid_combinations:
            return True

        if algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]:
            train_compatible = algorithm.training_type == dataset.training_type
        else:
            train_compatible = True
        return algorithm.input_dimensionality == dataset.input_dimensionality and train_compatible
