from .base import Adapter
from ..data_types import TSFunction, AlgorithmParameter


class FunctionAdapter(Adapter):

    def __init__(self, fn: TSFunction):
        self.fn = fn

    def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:
        return self.fn(dataset, args)

    @staticmethod
    def identity() -> 'FunctionAdapter':
        def identity_fn(data: AlgorithmParameter, _: dict) -> AlgorithmParameter:
            return data
        return FunctionAdapter(fn=identity_fn)
