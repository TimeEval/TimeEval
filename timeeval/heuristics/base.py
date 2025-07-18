import abc
import inspect
from pathlib import Path
from typing import Any, Dict, List

from timeeval import Algorithm
from timeeval.datasets import Dataset


class HeuristicFallbackWarning(UserWarning):
    """Warning that is raised if a heuristic falls back to a default value."""

    ...


class TimeEvalParameterHeuristic(abc.ABC):
    """Base class for TimeEval parameter heuristics.

    Heuristics are used to calculate parameter values for algorithms based on information about the algorithm, the
    dataset, or other parameters. They are evaluated in the driver process when TimeEval is configured. This means
    that the datasets must be available on the node executing the driver process. The calculated parameter values are
    then injected into the algorithm configuration and the algorithm is executed on the cluster.

    See Also
    --------
    :func:`timeeval.heuristics.inject_heuristic_values`
        Function that uses the heuristics to calculate parameter values for algorithms.
    """

    @abc.abstractmethod
    def __call__(self, algorithm: Algorithm, dataset_details: Dataset, dataset_path: Path, **kwargs) -> Any:  # type: ignore[no-untyped-def]
        """
        Calculate new parameter value based on information about algorithm, dataset metadata, and the dataset
        itself. If `None` is returned, the parameter is unset and the algorithm should use its default value.
        """
        ...

    @property
    def name(self) -> str:
        """Name of this parameter heuristic (corresponds to the class name)."""
        return self.__class__.__name__

    def parameters(self) -> Dict[str, Any]:
        """Get the heuristic's parameters (arguments to the heuristic) as a dictionary."""
        out = {}
        for key in self.get_param_names():
            out[key] = getattr(self, key)
        return out

    @classmethod
    def get_param_names(cls) -> List[str]:
        """
        Get parameter names (arguments) for the heuristic.

        Adopted from https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/base.py.
        """
        # fetch the constructor or the original constructor before deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "Heuristic implementations should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
