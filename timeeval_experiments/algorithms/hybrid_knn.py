from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for Hybrid-KNN
def post_hybrid_knn(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_hybrid_knn_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "windowing size for time series",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "batch_size": {
  "defaultValue": 64,
  "description": "number of simultaneously trained data instances",
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
  "description": "number of training iterations over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Gradient factor for backpropagation",
  "name": "learning_rate",
  "type": "float"
 },
 "linear_layer_shape": {
  "defaultValue": [
   100,
   10
  ],
  "description": "NN structure with embedding dim as last value",
  "name": "linear_layer_shape",
  "type": "List[int]"
 },
 "n_estimators": {
  "defaultValue": 3,
  "description": "Defines number of ensembles",
  "name": "n_estimators",
  "type": "int"
 },
 "n_neighbors": {
  "defaultValue": 12,
  "description": "Defines which neighbour's distance to use",
  "name": "n_neighbors",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "train-validation split",
  "name": "split",
  "type": "float"
 },
 "test_batch_size": {
  "defaultValue": 256,
  "description": "number of simultaneously tested data instances",
  "name": "test_batch_size",
  "type": "int"
 }
}


def hybrid_knn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Hybrid KNN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/hybrid_knn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_hybrid_knn,
        param_schema=_hybrid_knn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
