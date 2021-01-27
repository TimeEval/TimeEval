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
    hyper_parameters: dict

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        for k, v in dictionary.items():
            if isinstance(v, (PosixPath, WindowsPath)):
                dictionary[k] = str(v)
        return json.dumps(json.dumps(dictionary))


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, hyper_parameters: dict):
        self.image_name = image_name
        self.hyper_parameters = hyper_parameters
        self.client = docker.from_env()

    def _run_container(self, dataset_path: Path, args: dict):
        algorithm_interface = AlgorithmInterface(
            dataset_path=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            results_path=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
            hyper_parameters=self.hyper_parameters
        )

        self.client.containers.run(
            f"{self.image_name}:latest",
            f'--inputstring {algorithm_interface.to_json_string()}',
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path").absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            }
        )

    def _read_results(self, args: dict) -> np.ndarray:
        return np.loadtxt(args.get("results_path") / Path(SCORES_FILE_NAME))

    def _call(self, dataset: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        args = args or {}
        self._run_container(dataset, args)
        return self._read_results(args)
