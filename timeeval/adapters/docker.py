import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path, PurePath, PurePosixPath
from traceback import print_exc
from typing import Optional, Any, Callable, Tuple, Dict

import docker
import numpy as np
import requests
from docker.errors import DockerException, ImageNotFound, APIError
from docker.models.containers import Container
from durations import Duration
from numpyencoder import NumpyEncoder

from .base import Adapter, AlgorithmParameter
from ..data_types import ExecutionType
from ..resource_constraints import ResourceConstraints, GB

from ..utils.exceptions import exc_causes

DATASET_TARGET_PATH = PurePosixPath("/data")
RESULTS_TARGET_PATH = PurePosixPath("/results")
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "model.pkl"


class DockerJSONEncoder(NumpyEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExecutionType):
            return o.name.lower()
        elif isinstance(o, (Path, PurePath)):
            return str(o)
        return super().default(o)


class DockerAdapterInternalError(Exception):
    pass


class DockerTimeoutError(Exception):
    pass


class DockerMemoryError(Exception):
    pass


class DockerAlgorithmFailedError(Exception):
    pass


@dataclass
class AlgorithmInterface:
    dataInput: PurePath
    dataOutput: PurePath
    modelInput: PurePath
    modelOutput: PurePath
    executionType: ExecutionType
    customParameters: Dict[str, Any] = field(default_factory=dict)

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


class DockerAdapter(Adapter):
    """
    An adapter that allows to run a Docker image as an anomaly detector.
    You can find a list of available Docker images on `GitHub <https://github.com/TimeEval/TimeEval-algorithms>`_.

    Parameters
    ----------

    image_name : str
        The name of the Docker image to run.

    tag : str
        The tag of the Docker image to run. Defaults to "latest".

    group_privileges : str
        The group privileges to use for the Docker container. Defaults to "akita".

    skip_pull : bool
        Whether to skip pulling the Docker image. Defaults to False.

    timeout : Optional[Duration]
        The timeout for the Docker container. If not set, the timeout is taken from the :class:`~timeeval.resource_contraints.ResourceConstraints`.

    memory_limit_overwrite : Optional[int]
        The memory limit for the Docker container. If not set, the memory limit is taken from the :class:`~timeeval.resource_contraints.ResourceConstraints`.

    cpu_limit_overwrite : Optional[float]
        The CPU limit for the Docker container. If not set, the CPU limit is taken from the :class:`~timeeval.resource_contraints.ResourceConstraints`.
    """
    def __init__(self, image_name: str, tag: str = "latest", group_privileges: str = "akita", skip_pull: bool = False,
                 timeout: Optional[Duration] = None, memory_limit_overwrite: Optional[int] = None,
                 cpu_limit_overwrite: Optional[float] = None) -> None:
        self.image_name = image_name
        self.tag = tag
        self.group = group_privileges
        self.skip_pull = skip_pull
        self.timeout = timeout
        self.memory_limit = memory_limit_overwrite
        self.cpu_limit = cpu_limit_overwrite

    @staticmethod
    def _get_gid(group: str) -> str:
        if os.name == "nt":
            return ""

        CMD = "getent group %s | cut -d ':' -f 3"
        return subprocess.run(CMD % group, capture_output=True, text=True, shell=True).stdout.strip()

    @staticmethod
    def _get_uid() -> str:
        if os.name == "nt":
            return ""

        uid = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
        if uid == "0":  # if uid is root (0), we don't want to change it
            return ""
        else:
            return uid

    def _prepare_env(self) -> Dict[str, str]:
        uid = DockerAdapter._get_uid()
        gid = DockerAdapter._get_gid(self.group)
        if uid and not gid:
            gid = uid
        env = {}
        if uid:
            env["LOCAL_UID"] = uid
        if gid:
            env["LOCAL_GID"] = gid
        return env

    def _get_compute_limits(self, args: Dict[str, Any]) -> Tuple[int, float]:
        limits: Tuple[int, float] = args.get("resource_constraints", ResourceConstraints()).get_compute_resource_limits(
            memory_overwrite=self.memory_limit,
            cpu_overwrite=self.cpu_limit
        )
        return limits

    def _get_timeout(self, args: Dict[str, Any]) -> Duration:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        if exec_type == ExecutionType.TRAIN or exec_type == ExecutionType.TRAIN.value:
            return constraints.get_train_timeout(self.timeout)
        else:
            return constraints.get_execute_timeout(self.timeout)

    @staticmethod
    def _should_use_prelim_model(args: Dict[str, Any]) -> bool:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        result: bool = (exec_type == ExecutionType.TRAIN or exec_type == ExecutionType.TRAIN.value) and constraints.use_preliminary_model_on_train_timeout
        return result

    @staticmethod
    def _should_use_prelim_results(args: Dict[str, Any]) -> bool:
        exec_type = args.get("executionType", "")
        constraints = args.get("resource_constraints", ResourceConstraints())
        result: bool = (exec_type == ExecutionType.EXECUTE or exec_type == ExecutionType.EXECUTE.value) and constraints.use_preliminary_scores_on_execute_timeout
        return result

    @staticmethod
    def _results_path(args: Dict[str, Any], absolute: bool = False) -> Path:
        path: Path = args.get("results_path", Path("./results"))
        if absolute:
            path = path.resolve()
        return path

    def _run_container(self, dataset_path: Path, args: Dict[str, Any]) -> Container:
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=DATASET_TARGET_PATH / dataset_path.name,
            dataOutput=RESULTS_TARGET_PATH / SCORES_FILE_NAME,
            modelInput=RESULTS_TARGET_PATH / MODEL_FILE_NAME,
            modelOutput=RESULTS_TARGET_PATH / MODEL_FILE_NAME,
            executionType=args.get("executionType", ExecutionType.EXECUTE.value),
            customParameters=args.get("hyper_params", {}),
        )
        env_vars = self._prepare_env()
        print(f"Running container '{self.image_name}:{self.tag}' with env='{repr(env_vars)}' in {algorithm_interface.executionType} mode.")

        memory_limit, cpu_limit = self._get_compute_limits(args)
        cpu_shares = int(cpu_limit * 1e9)
        print(f"Restricting container to {cpu_limit} CPUs and {memory_limit / GB:.3f} GB RAM")

        try:
            return client.containers.run(
                f"{self.image_name}:{self.tag}",
                f"execute-algorithm '{algorithm_interface.to_json_string()}'",
                volumes={
                    str(dataset_path.parent.resolve()): {"bind": str(DATASET_TARGET_PATH), "mode": "ro"},
                    str(self._results_path(args, absolute=True)): {"bind": str(RESULTS_TARGET_PATH), "mode": "rw"}
                },
                environment=env_vars,
                mem_swappiness=0,
                mem_limit=memory_limit,
                memswap_limit=memory_limit,
                nano_cpus=cpu_shares,
                detach=True,
            )
        except (APIError, ImageNotFound) as e:
            reason = str(e)
            for exc in exc_causes(e):
                if type(exc) == ImageNotFound:
                    reason = "image not found"
                    break

            print(f"Could not start Docker container for algorithm ({self.image_name}:{self.tag}:")
            print_exc(file=sys.stdout)
            raise DockerAdapterInternalError(
                f"Could not start Docker container for algorithm {self.image_name}:{self.tag} because {reason}!"
            ) from None  # hides exception chain for driver process

    def _run_until_timeout(self, container: Container, args: Dict[str, Any]) -> None:
        timeout = self._get_timeout(args)
        try:
            result = container.wait(timeout=timeout.to_seconds())
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if "timed out" in str(e):
                if self._should_use_prelim_results(args):
                    # check whether results file is stored
                    if (self._results_path(args) / SCORES_FILE_NAME).is_file():
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
                    if (self._results_path(args) / MODEL_FILE_NAME).is_file():
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

        if result["StatusCode"] == 137:
            print(f"Docker algorithm ran out of memory (status {result['StatusCode']})!")
            raise DockerMemoryError(f"Docker algorithm exceeded memory limit of {self._get_compute_limits(args)[0]} Bytes!")

        elif result["StatusCode"] != 0:
            print(f"Docker algorithm failed with status code '{result['StatusCode']}', consider container logs above.")
            raise DockerAlgorithmFailedError(f"Status '{result['StatusCode']}', please consider log files in {self._results_path(args, absolute=True)}!")

    def _read_results(self, args: Dict[str, Any]) -> np.ndarray:
        results: np.ndarray = np.genfromtxt(self._results_path(args) / SCORES_FILE_NAME, delimiter=",")
        return results

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        assert isinstance(dataset, Path), \
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

            def prepare() -> None:
                client = docker.from_env(timeout=Duration("5 minutes").to_seconds())
                client.images.pull(image, tag=tag)

            return prepare
        else:
            return None

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        def finalize() -> None:
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
