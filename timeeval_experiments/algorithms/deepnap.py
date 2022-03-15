from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for DeepNAP
def post_deepnap(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 15)
    return ReverseWindowing(window_size=window_size * 2).fit_transform(scores)


_deepnap_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 15,
  "description": "Size of the sliding windows",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "batch_size": {
  "defaultValue": 32,
  "description": "Number of instances trained at the same time",
  "name": "batch_size",
  "type": "int"
 },
 "dropout": {
  "defaultValue": 0.5,
  "description": "Probability for a neuron to be zeroed for regularization",
  "name": "dropout",
  "type": "float"
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
  "description": "Number of training iterations over entire dataset; recommended value: 256",
  "name": "epochs",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "linear_hidden_size": {
  "defaultValue": 100,
  "description": "Number of neurons in linear hidden layer",
  "name": "linear_hidden_size",
  "type": "int"
 },
 "lstm_layers": {
  "defaultValue": 2,
  "description": "Number of LSTM layers within encoder and decoder",
  "name": "lstm_layers",
  "type": "int"
 },
 "partial_sequence_length": {
  "defaultValue": 3,
  "description": "Number of points taken from the beginning of the predicted window used to build a partial sequence (with neighboring points) that is passed through another linear network.",
  "name": "partial_sequence_length",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "rnn_hidden_size": {
  "defaultValue": 200,
  "description": "Number of neurons in LSTM hidden layer",
  "name": "rnn_hidden_size",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "validation_batch_size": {
  "defaultValue": 256,
  "description": "Number of instances used for validation at the same time",
  "name": "validation_batch_size",
  "type": "int"
 }
}


def deepnap(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DeepNAP",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/deepnap",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_deepnap,
        param_schema=_deepnap_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
