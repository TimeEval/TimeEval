from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_bagel_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 128,
  "description": "Batch size for input data",
  "name": "batch_size",
  "type": "int"
 },
 "cuda": {
  "defaultValue": False,
  "description": "Use GPU for training",
  "name": "cuda",
  "type": "boolean"
 },
 "dropout": {
  "defaultValue": 0.1,
  "description": "Rate of conditional dropout used",
  "name": "dropout",
  "type": "float"
 },
 "early_stopping_delta": {
  "defaultValue": 0.5,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 50,
  "description": "Number of passes over the entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "hidden_layer_shape": {
  "defaultValue": [
   100,
   100
  ],
  "description": "NN hidden layers structure",
  "name": "hidden_layer_shape",
  "type": "List[int]"
 },
 "latent_size": {
  "defaultValue": 8,
  "description": "Dimensionality of encoding",
  "name": "latent_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Fraction to split training data by for validation",
  "name": "split",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 120,
  "description": "Size of sliding windows",
  "name": "window_size",
  "type": "int"
 }
}


def bagel(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Bagel",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/bagel",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_bagel_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
