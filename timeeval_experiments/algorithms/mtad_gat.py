from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for MTAD-GAT
def post_mtad_gat(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 20)
    return ReverseWindowing(window_size=window_size + 1).fit_transform(scores)


_mtad_gat_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 64,
  "description": "Number of data points propagated in parallel",
  "name": "batch_size",
  "type": "int"
 },
 "context_window_size": {
  "defaultValue": 5,
  "description": "Window for mean in SR cleaning",
  "name": "context_window_size",
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
  "description": "Number of times the algorithm trains on the dataset",
  "name": "epochs",
  "type": "int"
 },
 "gamma": {
  "defaultValue": 0.8,
  "description": "Importance factor for posterior in scoring",
  "name": "gamma",
  "type": "float"
 },
 "kernel_size": {
  "defaultValue": 7,
  "description": "Kernel size for 1D-convolution",
  "name": "kernel_size",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 300,
  "description": "Embedding size in VAE",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Learning rate for training",
  "name": "learning_rate",
  "type": "float"
 },
 "linear_layer_shape": {
  "defaultValue": [
   300,
   300,
   300
  ],
  "description": "Architecture of FC-NN",
  "name": "linear_layer_shape",
  "type": "List[int]"
 },
 "mag_window_size": {
  "defaultValue": 3,
  "description": "Window size for sliding window average calculation",
  "name": "mag_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "score_window_size": {
  "defaultValue": 40,
  "description": "Window size for anomaly scoring",
  "name": "score_window_size",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "threshold": {
  "defaultValue": 3,
  "description": "Threshold for SR cleaning",
  "name": "threshold",
  "type": "float"
 },
 "window_size": {
  "defaultValue": 20,
  "description": "Window size for windowing of Time Series",
  "name": "window_size",
  "type": "int"
 }
}


def mtad_gat(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="MTAD-GAT",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/mtad_gat",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_mtad_gat,
        param_schema=_mtad_gat_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
