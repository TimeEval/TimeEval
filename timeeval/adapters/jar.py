import subprocess
from typing import Any, List, Dict

import numpy as np

from .base import Adapter
from ..data_types import AlgorithmParameter


class JarAdapter(Adapter):
    """
    An adapter that allows to run a jar file as an anomaly detector.

    .. warning::
        This adapter is deprecated and will be removed in a future version of TimeEval.

    Parameters
    ----------

    jar_file : str
        The path to the jar file to run.

    output_file : str
        The path to the file to which the jar file writes its output.

    args : List[Any]
        The arguments to pass to the jar file.

    kwargs : Dict[str, Any]
        The keyword arguments to pass to the jar file.

    verbose : bool
        Whether to print the output of the jar file to the console.
    """
    def __init__(self, jar_file: str, output_file: str, args: List[Any], kwargs: Dict[str, Any], verbose: bool = False) -> None:
        self.jar_file = jar_file
        self.output_file = output_file
        self.args = args
        self.kwargs = kwargs
        self.verbose = verbose

    def _format_args(self) -> str:
        return " ".join(self.args)

    def _format_kwargs(self) -> str:
        return " ".join(map(lambda x: f"--{x[0]} {x[1]}", self.kwargs.items()))

    def _read_results(self) -> np.ndarray:
        results: np.ndarray = np.loadtxt(self.output_file)
        return results

    def _call(self, _1: AlgorithmParameter, _2: Dict[str, Any]) -> np.ndarray:
        stdout = subprocess.STDOUT if self.verbose else subprocess.DEVNULL
        subprocess.call(f"java -jar {self.jar_file} {self._format_args()} {self._format_kwargs()}".split(),
                        stdout=stdout, stderr=subprocess.STDOUT)
        return self._read_results()
