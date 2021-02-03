import json
import subprocess
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
MODEL_FILE_NAME = "model.pkl"


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
    modelInput: Path
    modelOutput: Path
    customParameters: dict = field(default_factory=dict)
    executionType: ExecutionType = ExecutionType.EXECUTE

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


class DockerAdapter(BaseAdapter):
    def __init__(self, image_name: str, tag: str = "latest", group_privileges="akita", skip_pull=False):
        self.image_name = image_name
        self.tag = tag
        self.group = group_privileges
        self.skip_pull = skip_pull

    @staticmethod
    def _get_gid(group: str) -> str:
        CMD = "getent group %s | cut -d ':' -f 3"
        return subprocess.run(CMD % group, capture_output=True, text=True, shell=True).stdout.strip()

    @staticmethod
    def _get_uid() -> str:
        return subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()

    def _run_container(self, dataset_path: Path, args: dict):
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
            modelInput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            modelOutput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute()
        )

        gid = DockerAdapter._get_gid(self.group)
        uid = DockerAdapter._get_uid()
        client.containers.run(
            f"{self.image_name}:{self.tag}",
            f"execute-algorithm '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path", Path("./results")).absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            },
            environment={
                "LOCAL_GID": gid,
                "LOCAL_UID": uid
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

    def prepare(self):
        client = docker.from_env()
        if not self.skip_pull:
            client.images.pull(self.image_name, tag=self.tag)

    def prune(self):
        client = docker.from_env()
        client.containers.prune()
