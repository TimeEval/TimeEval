from durations import Duration
from sklearn.model_selection import ParameterGrid
from typing import Any, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for OmniAnomaly
def post_omni_anomaly(scores: np.ndarray, args: dict) -> np.ndarray:
    window_length = args.get("hyper_params", {}).get("window_length", 100)
    return ReverseWindowing(window_size=window_length).fit_transform(scores)


_omnianomaly_parameters = {
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
 "window_size": {
  "defaultValue": 100,
  "description": "Sliding window size",
  "name": "window_size",
  "type": "int"
 }
}


def omnianomaly(params: Any = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="OmniAnomaly",
        main=DockerAdapter(
            image_name="mut:5000/akita/omnianomaly",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_omni_anomaly,
        params=_omnianomaly_parameters,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
