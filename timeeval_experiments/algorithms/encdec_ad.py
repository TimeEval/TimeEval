from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for EncDec-AD
def post_encdec_ad(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    return ReverseWindowing(window_size=2 * window_size).fit_transform(scores)


_encdec_ad_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 30,
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
 "latent_size": {
  "defaultValue": 40,
  "description": "Size of the autoencoder's latent space (embedding size)",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "lstm_layers": {
  "defaultValue": 1,
  "description": "Number of LSTM layers within encoder and decoder",
  "name": "lstm_layers",
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
  "description": "Size of the sliding windows",
  "name": "window_size",
  "type": "int"
 }
}


def encdec_ad(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="EncDec-AD",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/encdec_ad",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_encdec_ad,
        param_schema=_encdec_ad_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
