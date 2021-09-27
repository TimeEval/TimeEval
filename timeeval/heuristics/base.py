import abc
import inspect
from pathlib import Path
from typing import Any, Dict

from timeeval import Algorithm
from timeeval.datasets import Dataset


class TimeEvalParameterHeuristic(abc.ABC):
    @abc.abstractmethod
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Any:
        """
        Calculate new parameter value based on information about algorithm, dataset metadata, and the dataset
        itself. If `None` is returned, the parameter is unset and the algorithm should use its default value.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> Dict[str, Any]:
        out = {}
        for key in self.get_param_names():
            out[key] = getattr(self, key)
        return out

    @classmethod
    def get_param_names(cls):
        """
        Get parameter names for the heuristic

        Adopted from https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/base.py
        """
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("Heuristic implementations should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
