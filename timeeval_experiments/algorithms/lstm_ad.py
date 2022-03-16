from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for LSTM-AD
def post_lstm_ad(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 30)
    prediction_window_size = args.get("hyper_params", {}).get("prediction_window_size", 1)
    return ReverseWindowing(window_size=window_size + prediction_window_size).fit_transform(scores)


_lstm_ad_parameters: Dict[str, Dict[str, Any]] = {
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
  "defaultValue": 50,
  "description": "Number of training iterations over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "lstm_layers": {
  "defaultValue": 2,
  "description": "Number of stacked LSTM layers",
  "name": "lstm_layers",
  "type": "int"
 },
 "prediction_window_size": {
  "defaultValue": 1,
  "description": "Number of points predicted",
  "name": "prediction_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.9,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "test_batch_size": {
  "defaultValue": 128,
  "description": "Number of instances used for testing at the same time",
  "name": "test_batch_size",
  "type": "int"
 },
 "validation_batch_size": {
  "defaultValue": 128,
  "description": "Number of instances used for validation at the same time",
  "name": "validation_batch_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 30,
  "description": "",
  "name": "window_size",
  "type": "int"
 }
}


def lstm_ad(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LSTM-AD",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/lstm_ad",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_lstm_ad,
        param_schema=_lstm_ad_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
