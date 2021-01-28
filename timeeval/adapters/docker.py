import numpy as np
from typing import Union, Optional
from pathlib import Path, WindowsPath, PosixPath
import docker
from docker.models.containers import Container
from dataclasses import dataclass, asdict
import json

from .base import BaseAdapter, AlgorithmParameter


DATASET_TARGET_PATH = "/data/"
RESULTS_TARGET_PATH = "/results"
SCORES_FILE_NAME = "anomaly_scores.ts"


@dataclass
class AlgorithmInterface:
    dataset_path: Path
    results_path: Path

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        for k, v in dictionary.items():
            if isinstance(v, (PosixPath, WindowsPath)):
                dictionary[k] = str(v)
        return json.dumps(dictionary)


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, tag: str = "latest"):
        self.image_name = image_name
        self.tag = tag
        self.client = docker.from_env()

    def _run_container(self, dataset_path: Path, args: dict):
        algorithm_interface = AlgorithmInterface(
            dataset_path=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            results_path=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute()
        )

        self.client.containers.run(
            f"{self.image_name}:{self.tag}",
            f"--inputstring '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path", Path("./results")).absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            }
        )

    def _read_results(self, args: dict) -> np.ndarray:
        return np.loadtxt(args.get("results_path", Path("./results")) / Path(SCORES_FILE_NAME))

    def _call(self, dataset: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        args = args or {}
        self._run_container(dataset, args)
        return self._read_results(args)

    def make_available(self):
        self.client.images.pull(self.image_name, tag=self.tag)
