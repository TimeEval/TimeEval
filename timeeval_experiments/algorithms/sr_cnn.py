from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_sr_cnn_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 256,
  "description": "Number of data points trained in parallel",
  "name": "batch_size",
  "type": "int"
 },
 "early_stopping_delta": {
  "defaultValue": 0.05,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 1,
  "description": "Number of training passes over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 1e-06,
  "description": "Gradient factor during SGD training",
  "name": "learning_rate",
  "type": "float"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "Number of processes used during training",
  "name": "n_jobs",
  "type": "int"
 },
 "num": {
  "defaultValue": 10,
  "description": "Max value for generated data",
  "name": "num",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generators",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.9,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "step": {
  "defaultValue": 64,
  "description": "stride size for training data generation",
  "name": "step",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 128,
  "description": "Sliding window size",
  "name": "window_size",
  "type": "int"
 }
}


def sr_cnn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="SR-CNN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/sr_cnn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_sr_cnn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
