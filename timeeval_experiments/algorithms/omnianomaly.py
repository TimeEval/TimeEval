from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for OmniAnomaly
def post_omni_anomaly(scores: np.ndarray, args: dict) -> np.ndarray:
    window_length = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_length).fit_transform(scores)


_omnianomaly_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 50,
  "description": "Number of datapoints fitted parallel",
  "name": "batch_size",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 10,
  "description": "Number of training passes over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "l2_reg": {
  "defaultValue": 0.0001,
  "description": "Regularization factor",
  "name": "l2_reg",
  "type": "float"
 },
 "latent_size": {
  "defaultValue": 3,
  "description": "Reduced dimension size",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning Rate for Adam Optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "linear_hidden_size": {
  "defaultValue": 500,
  "description": "Dense layer size",
  "name": "linear_hidden_size",
  "type": "int"
 },
 "nf_layers": {
  "defaultValue": 20,
  "description": "NF layer size",
  "name": "nf_layers",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "rnn_hidden_size": {
  "defaultValue": 500,
  "description": "Size of RNN hidden layer",
  "name": "rnn_hidden_size",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split",
  "name": "split",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 100,
  "description": "Sliding window size",
  "name": "window_size",
  "type": "int"
 }
}


def omnianomaly(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="OmniAnomaly",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/omnianomaly",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_omni_anomaly,
        param_schema=_omnianomaly_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
