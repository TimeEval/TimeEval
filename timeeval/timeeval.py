import numpy as np
from typing import List, Callable
from collections import defaultdict
import tqdm

from timeeval.utils.metrics import roc


class TimeEval:
    def __init__(self,
                 datasets: List[str],
                 algorithms: List[Callable],
                 prepare_data: Callable[[np.ndarray], np.ndarray]):
        self.dataset_names = datasets
        self.algorithms = algorithms
        self.datasets: List[np.ndarray] = []
        self.prepare_data = prepare_data

        self.results = defaultdict(dict)

    def _load_dataset(self, name) -> np.ndarray:
        pass

    def _load_datasets(self):
        for name in tqdm.tqdm(self.dataset_names, desc="Load datasets"):
            dataset = self._load_dataset(name)
            dataset = self.prepare_data(dataset)
            self.datasets.append(dataset)

    def _run_algorithm(self, algorithm: Callable):
        assert len(self.datasets) > 0, "No datasets loaded for evaluation"

        for dataset_name, dataset in tqdm.tqdm(zip(self.dataset_names, self.datasets), desc="Datasets", position=1):
            y_true = dataset[:, 1]
            y_scores = algorithm(dataset[:, 0])
            self.results[str(algorithm)][dataset_name] = roc(y_scores, y_true, plot=False)

    def run(self):
        assert len(self.algorithms) > 0, "No algorithms given for evaluation"

        self._load_datasets()
        # todo: possibility to run multiple algorithms in parallel
        for algorithm in tqdm.tqdm(self.algorithms, desc="Algorithms", position=0):
            self._run_algorithm(algorithm)
