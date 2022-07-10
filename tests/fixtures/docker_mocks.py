from pathlib import Path
from typing import List

import numpy as np
from docker.models.containers import Container

from timeeval.adapters.docker import SCORES_FILE_NAME


TEST_DOCKER_IMAGE = "registry.gitlab.hpi.de/akita/i/timeeval-test-algorithm"


class MockDockerContainer:
    def __init__(self, write_scores_file: bool = False):
        self.stopped = True
        self._write_scores_file = write_scores_file

    def wait(self, timeout=None) -> dict:
        return {"Error": None, "StatusCode": 0}

    def run(self, image: str, cmd: str, volumes: dict, **kwargs):
        self.stopped = False
        self.image = image
        self.cmd = cmd
        self.volumes = volumes
        self.run_kwargs = kwargs

        real_path = Path(list(volumes.items())[1][0]).resolve()
        if self._write_scores_file:
            np.arange(10, dtype=np.float64).tofile(real_path / SCORES_FILE_NAME, sep="\n")
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
    def __init__(self, write_scores_file: bool = False):
        self.containers = MockDockerContainer(write_scores_file)
        self.images = MockImages()
