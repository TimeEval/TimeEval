from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


import numpy as np
from timeeval.utils.window import ReverseWindowing
# post-processing for MSCRED
def post_mscred(scores: np.ndarray, args: dict) -> np.ndarray:
    ds_length = args.get("dataset_details").length  # type: ignore
    gap_time = args.get("hyper_params", {}).get("gap_time", 10)
    window_size = args.get("hyper_params", {}).get("window_size", 5)
    max_window_size = max(args.get("hyper_params", {}).get("windows", [10, 30, 60]))
    offset = (ds_length - (max_window_size - 1)) % gap_time
    image_scores = ReverseWindowing(window_size=window_size).fit_transform(scores)
    return np.concatenate([np.repeat(image_scores[:-offset], gap_time), image_scores[-offset:]])


_mscred_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 32,
  "description": "Number of instances trained at the same time",
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
  "description": "Number of training iterations over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "gap_time": {
  "defaultValue": 10,
  "description": "Number of points to skip over between the generation of signature matrices",
  "name": "gap_time",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "test_batch_size": {
  "defaultValue": 256,
  "description": "Number of instances used for validation and testing at the same time",
  "name": "test_batch_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 5,
  "description": "Size of the sliding windows",
  "name": "window_size",
  "type": "int"
 },
 "windows": {
  "defaultValue": [
   10,
   30,
   60
  ],
  "description": "Number and size of different signature matrices (correlation matrices) to compute as a preprocessing step",
  "name": "windows",
  "type": "List[int]"
 }
}


def mscred(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="MSCRED",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/mscred",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_mscred,
        param_schema=_mscred_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
