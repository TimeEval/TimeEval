from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

from timeeval.adapters.base import Adapter
from timeeval.data_types import (
    TSFunction, TSFunctionPost, ExecutionType, AlgorithmParameter, TrainingType, InputDimensionality
)
from timeeval.params import ParameterConfig


@dataclass
class Algorithm:
    """
    This class is a wrapper for any `Adapter` and an instruction plan for the TimeEval tool.
    It tells TimeEval what algorithm to execute, what pre- and post-steps to perform
    and how the parameters and data are provided to the algorithm.
    Moreover, it defines attributes that are necessary to help TimeEval know
    what kind of time series can be put into the algorithm.

    Parameter
    ---------
    name : str
        The name of the Algorithm shown in the results.
    main : Adapter
        Adapter The [Adapter](timeeval/adapters/base.py) that contains the algorithm to evaluate.
    preprocess : Optional[TSFunction]
        Optional function to perform before `main` to modify input data.
    postprocess : Optional[TSFunctionPost]
        Optional function to perform after `main` to modify output data.
    data_as_file : bool
        Whether the data is input as `Path` or as `numpy.ndarray`.
    param_schema : Dict[str, Dict[str, Any]]
        Optional schema of the algorithm's input parameters needed by AlgorithmConfigurator. Schema definition::
            [
                "param_name": {
                    "name": str
                    "defaultValue": Any
                    "description": str
                    "type": str
                },
            ]
    param_config : ParameterConfig
        Optional object of type ParameterConfig to define a search grid or fixed parameters.
    training_type : TrainingType
        Definition of training type to receive the correct dataset formats (needed if TimeEval is run with
        `force_training_type_match` config).
    input_dimensionality : InputDimensionality
        Definition of training type to receive the correct dataset formats (needed if TimeEval is run with
        `force_dimensionality_match` config).
    """

    name: str
    main: Adapter
    preprocess: Optional[TSFunction] = None
    postprocess: Optional[TSFunctionPost] = None
    data_as_file: bool = False
    param_schema: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {})
    param_config: ParameterConfig = ParameterConfig.defaults()
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
        fn = self.prepare_fn()
        if fn:
            fn()

    def finalize_fn(self) -> Optional[Callable[[], None]]:
        return self.main.get_finalize_fn()

    def finalize(self) -> None:
        fn = self.finalize_fn()
        if fn:
            fn()
