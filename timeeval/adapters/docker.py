import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path, WindowsPath, PosixPath
from typing import Union, Optional, Any

import docker
import numpy as np

from .base import BaseAdapter, AlgorithmParameter

DATASET_TARGET_PATH = "/data/"
RESULTS_TARGET_PATH = "/results"
SCORES_FILE_NAME = "anomaly_scores.ts"


class DockerJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExecutionType):
            return o.name.lower()
        elif isinstance(o, (PosixPath, WindowsPath)):
            return str(o)
        return super().default(o)


class ExecutionType(Enum):
    TRAIN = 0
    EXECUTE = 1


@dataclass
class AlgorithmInterface:
    dataInput: Path
    dataOutput: Path
    customParameters: dict = field(default_factory=dict)
    executionType: ExecutionType = ExecutionType.EXECUTE
    modelInput: Optional[Path] = None
    modelOutput: Optional[Path] = None

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str):
        self.image_name = image_name

    def _run_container(self, dataset_path: Path, args: dict):
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute()
        )

        client.containers.run(
            f"{self.image_name}:latest",
            f"execute-algorithm '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path", Path("./results")).absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            }
        )

    def _read_results(self, args: dict) -> np.ndarray:
        return np.loadtxt(args.get("results_path", Path("./results")) / SCORES_FILE_NAME)

    def _call(self, dataset: Union[np.ndarray, Path], args: Optional[dict] = None) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        args = args or {}
        self._run_container(dataset, args)
        return self._read_results(args)
