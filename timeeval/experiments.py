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

    @property
    def dataset_collection(self) -> str:
        return self.dataset[0]

    @property
    def dataset_name(self) -> str:
        return self.dataset[1]

    def results_path(self, base_dir: Path, timestamp: str) -> Path:
        params_dir = hash_dict(self.params)
        return (base_dir / timestamp / self.algorithm.name / params_dir / self.dataset_collection /
                self.dataset_name / str(self.repetition))


class Experiments:

    def __init__(self, datasets: List[Tuple[str, str]], algorithms: List[Algorithm], repetitions: int = 1):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.repetitions = repetitions

    def __iter__(self) -> Generator[Experiment, None, None]:
        return self._create_gen()

    def _create_gen(self) -> Generator[Experiment, None, None]:
        for algorithm in self.algorithms:
            for algorithm_config in algorithm.param_grid:
                for dataset_name in self.dataset_names:
                    for repetition in range(1, self.repetitions + 1):
                        yield Experiment(
                            algorithm=algorithm,
                            dataset=dataset_name,
                            params=algorithm_config,
                            repetition=repetition
                        )
