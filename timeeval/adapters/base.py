from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict

from ..data_types import AlgorithmParameter, ExecutionType


class Adapter(ABC):
    """
    The base class for all adapters. An adapter is a wrapper around an anomaly detection algorithm that allows to
    execute it in a standardized way with the TimeEval framework. A subclass of Adapter must implement the _call method
    that executes the algorithm and returns the results. Optionally, it can also implement the get_prepare_fn and
    get_finalize_fn methods that are called before and after the execution of the algorithm, respectively.
    """
    @abstractmethod
    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:  # pragma: no cover
        """
        This method runs the anomaly detection algorithm and returns the results.

        Parameters
        ----------

        dataset : AlgorithmParameter
            The dataset to run the algorithm on.

        args : Dict[str, Any]
            The arguments to pass to the algorithm and TimeEval-internal configuration options.
            TimeEval arguments that get passed to the :class:`~timeeval.adapters.Adapter`-implementation. Example:: 
  
                 { 
                     "hyper_params": {}, 
                     "results_path": Path("results"), 
                     "resource_constraints": ResourceConstraints(), 
                     "dataset_details": ... 
                 } 
        """
        ...

    def __call__(self, dataset: AlgorithmParameter, args: Optional[Dict[str, Any]] = None) -> AlgorithmParameter:
        args = args or {}
        if "executionType" not in args:
            args["executionType"] = ExecutionType.EXECUTE
        return self._call(dataset, args)

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        """This method is executed before all algorithms are run."""
        return None

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        """This method is executed after all algorithms are run."""
        return None
