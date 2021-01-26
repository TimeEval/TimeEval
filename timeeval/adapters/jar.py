import subprocess
from typing import Optional

import numpy as np

from .base import BaseAdapter


class JarAdapter(BaseAdapter):
    def __init__(self, jar_file: str, output_file: str, args: list, kwargs: dict, verbose: bool = False):
        self.jar_file = jar_file
        self.output_file = output_file
        self.args = args
        self.kwargs = kwargs
        self.verbose = verbose

    def _format_args(self):
        return " ".join(self.args)

    def _format_kwargs(self):
        return " ".join(map(lambda x: f"--{x[0]} {x[1]}", self.kwargs.items()))

    def _read_results(self) -> np.ndarray:
        return np.loadtxt(self.output_file)

    def _call(self, _: np.ndarray, args: Optional[dict] = None) -> np.ndarray:
        stdout = subprocess.STDOUT if self.verbose else subprocess.DEVNULL
        subprocess.call(f"java -jar {self.jar_file} {self._format_args()} {self._format_kwargs()}".split(),
                        stdout=stdout, stderr=subprocess.STDOUT)
        return self._read_results()
