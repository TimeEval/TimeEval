from dataclasses import dataclass
from typing import Optional, Callable

from sklearn.model_selection import ParameterGrid

from .adapters.base import Adapter
from .data_types import TSFunction, TSFunctionPost, ExecutionType, AlgorithmParameter, TrainingType, InputDimensionality


@dataclass
class Algorithm:
    name: str
    main: Adapter
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
    param_grid: ParameterGrid = ParameterGrid({})
    training_type: TrainingType = TrainingType.UNSUPERVISED
    input_dimensionality: InputDimensionality = InputDimensionality.UNIVARIATE

    def train(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        if self.training_type == TrainingType.UNSUPERVISED:
            raise ValueError("Calling 'train()' on an unsupervised algorithm is not supported! "
                             f"Algorithm '{self.name}' has training type '{self.training_type.value}', but you tried "
                             f"to execute the training step. Please check the algorithm configuration for {self.name}!")
        args = args or {}
        args["executionType"] = ExecutionType.TRAIN
        return self.main(dataset, args)

    def execute(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        args = args or {}
        args["executionType"] = ExecutionType.EXECUTE
        return self.main(dataset, args)

    def prepare_fn(self) -> Optional[Callable[[], None]]:
        return self.main.get_prepare_fn()

    def prepare(self) -> None:
        if fn := self.prepare_fn():
            fn()

    def finalize_fn(self) -> Optional[Callable[[], None]]:
        return self.main.get_finalize_fn()

    def finalize(self) -> None:
        if fn := self.finalize_fn():
            fn()
