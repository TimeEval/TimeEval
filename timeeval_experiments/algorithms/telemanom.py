from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for telemanom
def post_telemanom(scores: np.ndarray, args: dict) -> np.ndarray:
    size = args.get("hyper_params", {}).get("window_size", 250) + args.get("hyper_params", {}).get("prediction_window_size", 10)
    return ReverseWindowing(window_size=size + 1).fit_transform(scores)


_telemanom_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 70,
  "description": "number of values to evaluate in each batch",
  "name": "batch_size",
  "type": "Int"
 },
 "dropout": {
  "defaultValue": 0.3,
  "description": "LSTM dropout probability",
  "name": "dropout",
  "type": "Float"
 },
 "early_stopping_delta": {
  "defaultValue": 0.0003,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "Float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "Int"
 },
 "epochs": {
  "defaultValue": 35,
  "description": "Number of training iterations over entire dataset",
  "name": "epochs",
  "type": "Int"
 },
 "error_buffer": {
  "defaultValue": 100,
  "description": "number of values surrounding an error that are brought into the sequence (promotes grouping on nearby sequences)",
  "name": "error_buffer",
  "type": "Int"
 },
 "lstm_batch_size": {
  "defaultValue": 64,
  "description": "number of vlaues to evaluate in one batch for the LSTM",
  "name": "lstm_batch_size",
  "type": "Int"
 },
 "p": {
  "defaultValue": 0.13,
  "description": "minimum percent decrease between max errors in anomalous sequences (used for pruning)",
  "name": "p",
  "type": "Float"
 },
 "prediction_window_size": {
  "defaultValue": 10,
  "description": "number of steps to predict ahead",
  "name": "prediction_window_size",
  "type": "Int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "smoothing_perc": {
  "defaultValue": 0.05,
  "description": "etermines window size used in EWMA smoothing (percentage of total values for channel)",
  "name": "smoothing_perc",
  "type": "Float"
 },
 "smoothing_window_size": {
  "defaultValue": 30,
  "description": "number of trailing batches to use in error calculation",
  "name": "smoothing_window_size",
  "type": "Int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "Float"
 },
 "window_size": {
  "defaultValue": 250,
  "description": "num previous timesteps provided to model to predict future values",
  "name": "window_size",
  "type": "Int"
 }
}


def telemanom(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Telemanom",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/telemanom",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_telemanom,
        param_schema=_telemanom_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
