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
    input_file: Path
    hyper_parameters: dict


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, hyper_parameters: dict):
        self.image_name = image_name
        self.hyper_parameters = hyper_parameters
        self.client = docker.from_env()
        self.container: Optional[Container] = None

    def _run_container(self, dataset_path: Path, args: dict):
        input_json = {
            "dataset_path": str((Path(DATASET_TARGET_PATH) / dataset_path.name).absolute()),
            "results_path": str((Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute())
        }
        input_json.update(self.hyper_parameters)

        self.container = self.client.containers.run(
            f"{self.image_name}:latest",
            f'--inputstring {json.dumps(json.dumps(input_json))}',
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path", Path("./results")).absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            })

    def _read_results(self, args: dict) -> np.ndarray:
        return np.loadtxt(args.get("results_path", Path("./results")) / Path(SCORES_FILE_NAME))

    def _call(self, dataset: Union[np.ndarray, Path], args: dict) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        self._run_container(dataset, args)
        return self._read_results(args)
