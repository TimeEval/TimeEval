from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for TAnoGan
def post_tanogan(scores: np.ndarray, args: dict) -> np.ndarray:
    length = args.get("dataset_details").length  # type: ignore
    window_size = args.get("hyper_params", {}).get("window_size", 30)
    scores = np.repeat(scores, repeats=window_size)
    result = np.full(shape=length, fill_value=np.nan)
    result[:scores.shape[0]] = scores
    return result


_tanogan_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 32,
  "description": "Number of instances trained at the same time",
  "name": "batch_size",
  "type": "int"
 },
 "cuda": {
  "defaultValue": "False",
  "description": "Set to `True`, if the GPU-backend (using CUDA) should be used. Otherwise, the algorithm is executed on the CPU.",
  "name": "cuda",
  "type": "boolean"
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
  "description": "Number of training iterations over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "iterations": {
  "defaultValue": 25,
  "description": "Number of test iterations per window",
  "name": "iterations",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.0002,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "Number of workers (processes) used to load and preprocess the data",
  "name": "n_jobs",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 30,
  "description": "Size of the sliding windows",
  "name": "window_size",
  "type": "int"
 }
}


def tanogan(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="TAnoGan",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/tanogan",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_tanogan,
        param_schema=_tanogan_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
