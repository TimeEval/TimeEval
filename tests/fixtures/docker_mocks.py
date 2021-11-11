from pathlib import Path
from typing import List

import numpy as np
from docker.models.containers import Container

from timeeval.adapters.docker import SCORES_FILE_NAME


TEST_DOCKER_IMAGE = "mut:5000/akita/timeeval-test-algorithm"


class MockDockerContainer:
    def __init__(self):
        self.stopped = True

    def wait(self, timeout=None) -> dict:
        return {"Error": None, "StatusCode": 0}

    def run(self, image: str, cmd: str, volumes: dict, **kwargs):
        self.stopped = False
        self.image = image
        self.cmd = cmd
        self.volumes = volumes
        self.run_kwargs = kwargs

        real_path = list(volumes.items())[1][0]
        if real_path.startswith("/tmp"):
            np.arange(10, dtype=np.float64).tofile(real_path / Path(SCORES_FILE_NAME), sep="\n")
        return self

    def prune(self, *args, **kwargs) -> None:
        pass

    def remove(self, *args, **kwargs) -> None:
        pass

    def stop(self, *args, **kwargs) -> None:
        self.stopped = True

    def logs(self) -> bytes:
        return "".encode("utf-8")

    def list(self, *args, **kwargs) -> List[Container]:
        return [self]


class MockImages:
    def pull(self, image, tag):
        pass


class MockDockerClient:
    def __init__(self):
        self.containers = MockDockerContainer()
        self.images = MockImages()
