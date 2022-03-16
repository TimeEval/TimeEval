from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for Normalizing Flows
def post_nf(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_normalizing_flows_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 64,
  "description": "How many data instances are trained at the same time.",
  "name": "batch_size",
  "type": "int"
 },
 "distillation_iterations": {
  "defaultValue": 1,
  "description": "Number of training steps for distillation",
  "name": "distillation_iterations",
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
  "description": "Number of training epochs",
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
 "n_hidden_features_factor": {
  "defaultValue": 1.0,
  "description": "Factor deciding how many hidden features for NFs are used based on number of features",
  "name": "n_hidden_features_factor",
  "type": "float"
 },
 "percentile": {
  "defaultValue": 0.05,
  "description": "Percentile defining the tails for anomaly sampling.",
  "name": "percentile",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.9,
  "description": "Train-validation split",
  "name": "split",
  "type": "float"
 },
 "teacher_epochs": {
  "defaultValue": 1,
  "description": "Number of epochs for teacher NF training",
  "name": "teacher_epochs",
  "type": "int"
 },
 "test_batch_size": {
  "defaultValue": 128,
  "description": "How many data instances are tested at the same time.",
  "name": "test_batch_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 20,
  "description": "Window size of sliding window over time series",
  "name": "window_size",
  "type": "int"
 }
}


def normalizing_flows(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Normalizing Flows",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/normalizing_flows",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_nf,
        param_schema=_normalizing_flows_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
