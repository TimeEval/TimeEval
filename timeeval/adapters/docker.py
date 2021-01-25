import numpy as np
from typing import Union, Optional
from pathlib import Path, WindowsPath, PosixPath
import docker
from docker.models.containers import Container

from .base import BaseAdapter


DATASET = "/data/data.csv"
RESULTS = "/results"
ANOMALY_SCORES = "anomaly_scores.ts"


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, results_dir: Path, hyper_parameters: dict):
        self.image_name = image_name
        self.hyper_parameters = hyper_parameters
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.logs = ""
        self.results_dir = results_dir / Path(image_name)

    def _get_results_dir(self, dataset_path: Path) -> Path:
        # todo: let's think about structure
        return self.results_dir / dataset_path.parent.name

    def _run_container(self, dataset_path: Path):
        self.container = self.client.containers.run(f"{self.image_name}:latest", volumes={
            str(dataset_path.absolute()): {'bind': DATASET, 'mode': 'ro'},
            str(self._get_results_dir(dataset_path).absolute()): {'bind': RESULTS, 'mode': 'rw'}
        })

    def _read_results(self, dataset_path: Path) -> np.ndarray:
        return np.loadtxt(self._get_results_dir(dataset_path) / Path(ANOMALY_SCORES))

    def _call(self, dataset: Union[np.ndarray, Path]) -> Union[np.ndarray, Path]:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        self._run_container(dataset)
        return self._read_results(dataset)
