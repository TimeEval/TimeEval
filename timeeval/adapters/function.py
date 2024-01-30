from typing import Any, Dict

from .base import Adapter
from ..data_types import TSFunction, AlgorithmParameter


class FunctionAdapter(Adapter):
    """
    An adapter that allows to run a function as an anomaly detector.

    Parameters
    ----------

    fn : TSFunction
        The function to run.
    """
    def __init__(self, fn: TSFunction):
        self.fn = fn

    def _call(self, dataset: AlgorithmParameter, args: Dict[str, Any]) -> AlgorithmParameter:
        # extract hyper parameters and forward them to the function
        params = args.get("hyper_params", {})
        return self.fn(dataset, params)

    @staticmethod
    def identity() -> 'FunctionAdapter':
        def identity_fn(data: AlgorithmParameter, _: Dict[str, Any]) -> AlgorithmParameter:
            return data

        return FunctionAdapter(fn=identity_fn)
