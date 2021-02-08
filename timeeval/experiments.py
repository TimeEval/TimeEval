from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Generator

from .algorithm import Algorithm
from .utils.hash_dict import hash_dict


@dataclass
class Experiment:
    dataset: Tuple[str, str]
    algorithm: Algorithm
    params: dict
    repetition: int
    base_results_dir: Path

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
            "hyper_params": self.params
        }


class Experiments:

    def __init__(self, datasets: List[Tuple[str, str]], algorithms: List[Algorithm], base_result_path: Path,
                 repetitions: int = 1):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions
        self.base_result_path = base_result_path

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
                            base_results_dir=self.base_result_path
                        )

    def __len__(self) -> int:
        return sum([len(algo.param_grid) for algo in self.algorithms]) * len(self.dataset_names) * self.repetitions
