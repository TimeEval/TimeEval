import json
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path, WindowsPath, PosixPath
from typing import Optional, Any, Callable, Tuple, Dict

import docker
import numpy as np
import requests
from docker.errors import DockerException
from docker.models.containers import Container
from durations import Duration
from numpyencoder import NumpyEncoder

from .base import Adapter, AlgorithmParameter
from ..data_types import ExecutionType
from ..resource_constraints import ResourceConstraints, GB

DATASET_TARGET_PATH = "/data"
RESULTS_TARGET_PATH = "/results"
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "model.pkl"


class DockerJSONEncoder(NumpyEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExecutionType):
            return o.name.lower()
        elif isinstance(o, (PosixPath, WindowsPath)):
            return str(o)
        return super().default(o)


class DockerTimeoutError(Exception):
    pass


class DockerAlgorithmFailedError(Exception):
    pass


@dataclass
class AlgorithmInterface:
    dataInput: Path
    dataOutput: Path
    modelInput: Path
    modelOutput: Path
    executionType: ExecutionType
    customParameters: Dict[str, Any] = field(default_factory=dict)

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


class DockerAdapter(Adapter):
    def __init__(self, image_name: str, tag: str = "latest", group_privileges="akita", skip_pull=False,
                 timeout: Optional[Duration] = None, memory_limit_overwrite: Optional[int] = None,
                 cpu_limit_overwrite: Optional[float] = None):
        self.image_name = image_name
        self.tag = tag
        self.group = group_privileges
        self.skip_pull = skip_pull
        self.timeout = timeout
        self.memory_limit = memory_limit_overwrite
        self.cpu_limit = cpu_limit_overwrite

    @staticmethod
    def _get_gid(group: str) -> str:
        CMD = "getent group %s | cut -d ':' -f 3"
        return subprocess.run(CMD % group, capture_output=True, text=True, shell=True).stdout.strip()

    @staticmethod
    def _get_uid() -> str:
        uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
        if uid == "0":  # if uid is root (0), we don't want to change it
            return ""
        else:
            return uid

    def _get_compute_limits(self, args: dict) -> Tuple[int, float]:
        return args.get("resource_constraints", ResourceConstraints()).get_compute_resource_limits(
            memory_overwrite=self.memory_limit,
            cpu_overwrite=self.cpu_limit
        )

    def _get_timeout(self, args: dict) -> Duration:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        if exec_type == ExecutionType.TRAIN or exec_type == ExecutionType.TRAIN.value:
            return constraints.get_train_timeout(self.timeout)
        else:
            return constraints.get_execute_timeout(self.timeout)

    @staticmethod
    def _should_use_prelim_model(args: dict) -> bool:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        return (exec_type == ExecutionType.TRAIN or exec_type == ExecutionType.TRAIN.value) and constraints.use_preliminary_model_on_train_timeout

    @staticmethod
    def _should_use_prelim_results(args: dict) -> bool:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        return (exec_type == ExecutionType.EXECUTE or exec_type == ExecutionType.EXECUTE.value) and constraints.use_preliminary_scores_on_execute_timeout

    @staticmethod
    def _results_path(args: Dict[str, Any], absolute: bool = False) -> Path:
        path: Path = args.get("results_path", Path("./results"))
        if absolute:
            path = path.resolve()
        return path

    def _run_container(self, dataset_path: Path, args: dict) -> Container:
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
            modelInput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            modelOutput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            executionType=args.get("executionType", ExecutionType.EXECUTE.value),
            customParameters=args.get("hyper_params", {}),
        )

        uid = DockerAdapter._get_uid()
        gid = DockerAdapter._get_gid(self.group)
        if not gid:
            gid = uid
        print(f"Running container '{self.image_name}:{self.tag}' with uid={uid} and gid={gid} privileges in {algorithm_interface.executionType} mode.")

        memory_limit, cpu_limit = self._get_compute_limits(args)
        cpu_shares = int(cpu_limit * 1e9)
        print(f"Restricting container to {cpu_limit} CPUs and {memory_limit / GB:.3f} GB RAM")

        return client.containers.run(
            f"{self.image_name}:{self.tag}",
            f"execute-algorithm '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(self._results_path(args, absolute=True)): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            },
            environment={
                "LOCAL_GID": gid,
                "LOCAL_UID": uid
            },
            mem_swappiness=0,
            mem_limit=memory_limit,
            memswap_limit=memory_limit,
            nano_cpus=cpu_shares,
            detach=True,
        )

    def _run_until_timeout(self, container: Container, args: dict):
        timeout = self._get_timeout(args)
        try:
            result = container.wait(timeout=timeout.to_seconds())
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if "timed out" in str(e):
                if self._should_use_prelim_results(args):
                    # check whether results file is stored
                    target_path = args.get("results_path", Path("./results"))
                    if (target_path / SCORES_FILE_NAME).is_file():
                        print(f"Container timeout after {timeout}, but TimeEval disregards this because "
                              f"'ResourceConstraints.preliminary_results_on_timeout' is set to True."
                              f"\nWill be using preliminary results for evaluation.")
                        result = {"StatusCode": 0}
                    else:
                        print(f"Container timeout after {timeout} and "
                              f"'ResourceConstraints.preliminary_results_on_timeout' is set to True. However, the "
                              f"algorithm did not store a preliminary result; raising DockerTimeoutError anyway!")
                        raise DockerTimeoutError(f"{self.image_name} could not create results after {timeout}") from e
                elif self._should_use_prelim_model(args):
                    # check if model was stored
                    target_path = args.get("results_path", Path("./results"))
                    if (target_path / MODEL_FILE_NAME).is_file():
                        print(f"Container timeout after {timeout}, but TimeEval disregards this because "
                              "'ResourceConstraints.use_preliminary_model_on_train_timeout' is set to True.")
                        result = {"StatusCode": 0}
                    else:
                        print(f"Container timeout after {timeout} and 'ResourceConstraints.use_preliminary_model_on_train_timeout' is "
                              "set to True. However, the algorithm did not store a model; "
                              "raising DockerTimeoutError anyway!")
                        raise DockerTimeoutError(f"{self.image_name} could not build a model within {timeout}") from e
                else:
                    print(f"Container timeout after {timeout}, raising DockerTimeoutError!")
                    raise DockerTimeoutError(f"{self.image_name} timed out after {timeout}") from e
            else:
                print(f"Waiting for container failed with error: {e}")
                raise e
        finally:
            print("\n#### Docker container logs ####")
            print(container.logs().decode("utf-8"))
            print("###############################\n")
            container.stop()

        if result["StatusCode"] != 0:
            print(f"Docker algorithm failed with status code '{result['StatusCode']}', consider container logs below.")
            raise DockerAlgorithmFailedError(f"Please consider log files in {self._results_path(args, absolute=True)}!")

    def _read_results(self, args: dict) -> np.ndarray:
        return np.genfromtxt(self._results_path(args) / SCORES_FILE_NAME, delimiter=",")

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        container = self._run_container(dataset, args)
        self._run_until_timeout(container, args)

        if args.get("executionType", ExecutionType.EXECUTE) == ExecutionType.EXECUTE:
            return self._read_results(args)
        else:
            return dataset

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        if not self.skip_pull:
            # capture variables for the function closure
            image: str = self.image_name
            tag: str = self.tag

            def prepare():
                client = docker.from_env(timeout=Duration("5 minutes").to_seconds())
                client.images.pull(image, tag=tag)

            return prepare
        else:
            return None

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        def finalize():
            client = docker.from_env(timeout=Duration("10 minutes").to_seconds())
            try:
                containers = client.containers.list(all=True, filters={"ancestor": self.image_name})
                for c in containers:
                    # force removal and also remove associated volumes
                    c.remove(force=True, v=True)
            except DockerException:
                # container cleanup is not critical; allow failure
                pass

        return finalize
