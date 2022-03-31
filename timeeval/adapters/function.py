from .base import Adapter
from ..data_types import TSFunction, AlgorithmParameter


class FunctionAdapter(Adapter):

    def __init__(self, fn: TSFunction):
        self.fn = fn

    def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:
        # extract hyper parameters and forward them to the function
        params = args.get("hyper_params", {})
        return self.fn(dataset, params)

    @staticmethod
    def identity() -> 'FunctionAdapter':
        def identity_fn(data: AlgorithmParameter, _: dict) -> AlgorithmParameter:
            return data

        return FunctionAdapter(fn=identity_fn)
