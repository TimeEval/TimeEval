from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Iterator, Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .times import Times
from ..algorithm import Algorithm
from ..constants import EXECUTION_LOG, ANOMALY_SCORES_TS, METRICS_CSV, HYPER_PARAMETERS
from ..data_types import AlgorithmParameter, TrainingType, InputDimensionality
from ..datasets import Datasets, Dataset
from ..heuristics import inject_heuristic_values
from ..metrics import Metric, DefaultMetrics
from ..params import Params
from ..resource_constraints import ResourceConstraints
from ..utils.datasets import extract_features, load_dataset, load_labels_only
from ..utils.encode_params import dump_params, dumps_params
from ..utils.results_path import generate_experiment_path


@dataclass
class Experiment:
    dataset: Dataset
    algorithm: Algorithm
    params: Params
    params_id: str
    repetition: int
    base_results_dir: Path
    resource_constraints: ResourceConstraints
    metrics: List[Metric]
    resolved_train_dataset_path: Optional[Path]
    resolved_test_dataset_path: Path

    @property
    def name(self) -> str:
        return f"{self.algorithm.name}-{self.dataset.collection_name}-{self.dataset.name}-{self.params_id}-{self.repetition}"

    @property
    def dataset_collection(self) -> str:
        return self.dataset.collection_name

    @property
    def dataset_name(self) -> str:
        return self.dataset.name

    @property
    def results_path(self) -> Path:
        return generate_experiment_path(self.base_results_dir, self.algorithm.name, self.params_id,
                                        self.dataset_collection, self.dataset_name, self.repetition)

    def build_args(self) -> Dict[str, Any]:
        return {
            "results_path": self.results_path,
            "resource_constraints": self.resource_constraints,
            "hyper_params": self.params.to_dict(),
            "dataset_details": self.dataset
        }

    @staticmethod
    def scale_scores(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true, dtype=np.int_)
        y_scores = np.asarray(y_scores, dtype=np.float_)

        # mask NaNs and Infs
        mask = np.isinf(y_scores) | np.isneginf(y_scores) | np.isnan(y_scores)

        # scale all other scores to [0, 1]
        scores = y_scores[~mask]
        if scores.size != 0:
            if len(scores.shape) == 1:
                scores = scores.reshape(-1, 1)
            y_scores[~mask] = MinMaxScaler().fit_transform(scores).ravel()
        return y_true, y_scores

    def evaluate(self) -> Dict[str, Any]:
        """
        Using TimeEval distributed, this method is executed on the remote node.
        """
        # materialize and persist hyper parameters to disk
        self.params = self.params.materialize()
        dump_params(self.params, self.results_path / HYPER_PARAMETERS)
        hyper_params = dumps_params(self.params)  # must be loaded before assess() or fail() will be called!

        try:
            with (self.results_path / EXECUTION_LOG).open("a") as logs_file:
                print(f"Starting evaluation of experiment {self.name}\n=============================================\n",
                      file=logs_file)

            # perform training if necessary
            result = self._perform_training()

            # perform execution
            y_scores, execution_times = self._perform_execution()
            result.update(execution_times)
            # backup results to disk
            pd.DataFrame([result]).to_csv(self.results_path / METRICS_CSV, index=False)

            y_true = load_labels_only(self.resolved_test_dataset_path)
            y_true, y_scores = self.scale_scores(y_true, y_scores)
            # persist scores to disk
            y_scores.tofile(str(self.results_path / ANOMALY_SCORES_TS), sep="\n")

            with (self.results_path / EXECUTION_LOG).open("a") as logs_file:
                print(f"Scoring algorithm {self.algorithm.name} with {','.join([m.name for m in self.metrics])} metrics",
                      file=logs_file)

                # calculate quality metrics
                errors = 0
                last_exception = None
                for metric in self.metrics:
                    print(f"Calculating {metric.name}", file=logs_file)
                    try:
                        score = metric(y_true, y_scores)
                        result[metric.name] = score
                        print(f"  = {score}", file=logs_file)
                        logs_file.flush()
                    except Exception as e:
                        print(f"Exception while computing metric {metric}: {e}", file=logs_file)
                        errors += 1
                        if str(e):
                            last_exception = e
                        continue

            # write all results to disk (overwriting backup)
            pd.DataFrame([result]).to_csv(self.results_path / METRICS_CSV, index=False)

            # potentially update parameter search space
            self.params.assess(y_true, y_scores)

            # rethrow exception if no metric could be calculated
            if errors == len(self.metrics) and last_exception is not None:
                raise last_exception

        except Exception as e:
            # on any exception, tell the parameter search process that this trial failed
            self.params.fail()
            raise e

        # add hyperparameters to result
        result["hyper_params"] = hyper_params
        return result

    def _perform_training(self) -> Dict[str, Any]:
        if self.algorithm.training_type == TrainingType.UNSUPERVISED:
            return {}

        if not self.resolved_train_dataset_path:
            raise ValueError(f"No training dataset was provided. Algorithm cannot be trained!")

        if self.algorithm.data_as_file:
            X: AlgorithmParameter = self.resolved_train_dataset_path
        else:
            X = load_dataset(self.resolved_train_dataset_path).values

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing training for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            times = Times.from_train_algorithm(self.algorithm, X, self.build_args())
        return times.to_dict()

    def _perform_execution(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.algorithm.data_as_file:
            X: AlgorithmParameter = self.resolved_test_dataset_path
        else:
            dataset = load_dataset(self.resolved_test_dataset_path)
            if dataset.shape[1] >= 3:
                X = extract_features(dataset)
            else:
                raise ValueError(
                    f"Dataset '{self.resolved_test_dataset_path.name}' has a shape that was not expected: {dataset.shape}")

        with (self.results_path / EXECUTION_LOG).open("a") as logs_file, redirect_stdout(logs_file):
            print(f"Performing execution for {self.algorithm.training_type.name} algorithm {self.algorithm.name}")
            y_scores, times = Times.from_execute_algorithm(self.algorithm, X, self.build_args())
        return y_scores, times.to_dict()


class Experiments:
    def __init__(self,
                 dmgr: Datasets,
                 datasets: List[Dataset],
                 algorithms: List[Algorithm],
                 base_result_path: Path,
                 resource_constraints: ResourceConstraints = ResourceConstraints.default_constraints(),
                 repetitions: int = 1,
                 metrics: Optional[List[Metric]] = None,
                 skip_invalid_combinations: bool = False,
                 force_training_type_match: bool = False,
                 force_dimensionality_match: bool = False,
                 experiment_combinations_file: Optional[Path] = None):
        self.dmgr = dmgr
        self.datasets = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions
        self.base_result_path = base_result_path
        self.resource_constraints = resource_constraints
        self.metrics = metrics or DefaultMetrics.default_list()
        self.skip_invalid_combinations = skip_invalid_combinations or force_training_type_match or force_dimensionality_match
        self.force_training_type_match = force_training_type_match
        self.force_dimensionality_match = force_dimensionality_match
        self.experiment_combinations: Optional[pd.DataFrame] = pd.read_csv(experiment_combinations_file) if experiment_combinations_file else None
        if self.skip_invalid_combinations or self.experiment_combinations is not None:
            self._N: Optional[int] = None
        else:
            self._N = sum(
                [len(algo.param_config) for algo in self.algorithms]
            ) * len(self.datasets) * self.repetitions
        self._experiments: Optional[List[Experiment]] = None

    def _should_be_run(self, algorithm: Algorithm, dataset: Dataset, params_id: str) -> bool:
        return self.experiment_combinations is None or \
                not self.experiment_combinations[
                    (self.experiment_combinations.algorithm == algorithm.name) &
                    (self.experiment_combinations.collection == dataset.datasetId[0]) &
                    (self.experiment_combinations.dataset == dataset.datasetId[1]) &
                    (self.experiment_combinations.hyper_params_id == params_id)
                ].empty

    def materialize_experiments(self) -> Iterator[Experiment]:
        for algorithm in self.algorithms:
            for dataset in self.datasets:
                if self._check_compatible(dataset, algorithm):
                    for algorithm_config in algorithm.param_config.iter(algorithm, dataset):
                        test_path, train_path = self._resolve_dataset_paths(dataset, algorithm)
                        # create parameter hash before executing heuristics
                        # (they replace the parameter values, but we want to be able to group by original configuration)
                        params_id = algorithm_config.uid()
                        if self._should_be_run(algorithm, dataset, params_id):
                            params = inject_heuristic_values(algorithm_config, algorithm, dataset, test_path)
                            for repetition in range(1, self.repetitions + 1):
                                yield Experiment(
                                    algorithm=algorithm,
                                    dataset=dataset,
                                    params=params,
                                    params_id=params_id,
                                    repetition=repetition,
                                    base_results_dir=self.base_result_path,
                                    resource_constraints=self.resource_constraints,
                                    metrics=self.metrics,
                                    resolved_test_dataset_path=test_path,
                                    resolved_train_dataset_path=train_path
                                )

    def __iter__(self) -> Iterator[Experiment]:
        if self._experiments is None:
            self._experiments = list(self.materialize_experiments())
        return iter(self._experiments)

    def __len__(self) -> int:
        if self._N is None and self._experiments is not None:
            self._N = len(self._experiments)
        elif self._N is None:
            self._experiments = list(self.materialize_experiments())
            self._N = len(self._experiments)
        return self._N

    def _resolve_dataset_paths(self, dataset: Dataset, algorithm: Algorithm) -> Tuple[Path, Optional[Path]]:
        test_dataset_path = self.dmgr.get_dataset_path(dataset.datasetId, train=False)
        train_dataset_path: Optional[Path] = None
        if algorithm.training_type != TrainingType.UNSUPERVISED:
            try:
                train_dataset_path = self.dmgr.get_dataset_path(dataset.datasetId, train=True)
            except KeyError:
                pass
        return test_dataset_path, train_dataset_path

    def _check_compatible(self, dataset: Dataset, algorithm: Algorithm) -> bool:
        if not self.skip_invalid_combinations:
            return True

        if (self.force_training_type_match or
                algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]):
            train_compatible = algorithm.training_type == dataset.training_type
        else:
            train_compatible = True

        if self.force_dimensionality_match:
            dim_compatible = algorithm.input_dimensionality == dataset.input_dimensionality
        else:
            """
            m = multivariate, u = univariate
            algo | data | res
              u  |  u   | 1
              u  |  m   | 0 <-- not compatible
              m  |  u   | 1
              m  |  m   | 1
            """
            dim_compatible = not (algorithm.input_dimensionality == InputDimensionality.UNIVARIATE and dataset.input_dimensionality == InputDimensionality.MULTIVARIATE)
        return dim_compatible and train_compatible
