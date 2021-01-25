import numpy as np
from typing import Union, Optional
from pathlib import Path, WindowsPath, PosixPath
import docker
from docker.models.containers import Container

from .base import BaseAdapter


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, algorithms_dir: Path, hyper_parameters: dict):
        self.image_name = image_name
        self.hyper_parameters = hyper_parameters
        self.client = docker.from_env()
        self.container: Optional[Container] = None
        self.logs = ""
        self.algorithms_dir = algorithms_dir

    def _start_container(self, dataset_path: Path):
        self.container = self.client.containers.run(f"{self.image_name}:latest", dataset_path)

    def _stop_container(self):
        self.logs = self.container.logs()
        self.container.stop()

    def _read_results(self) -> np.ndarray:
        return np.loadtxt(self.algorithms_dir / Path(self.image_name))

    def _call(self, dataset: Union[np.ndarray, Path]) -> Union[np.ndarray, Path]:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        self._start_container(dataset)
        self._stop_container()
        return self._read_results()
