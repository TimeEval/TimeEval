from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for Torsk
def _post_torsk(scores: np.ndarray, args: dict) -> np.ndarray:
    pred_size = args.get("hyper_params", {}).get("prediction_window_size", 20)
    context_window_size = args.get("hyper_params", {}).get("context_window_size", 10)
    size = pred_size * context_window_size + 1
    return ReverseWindowing(window_size=size).fit_transform(scores)


_torsk_parameters: Dict[str, Dict[str, Any]] = {
 "context_window_size": {
  "defaultValue": 10,
  "description": "Size of a tumbling window used to encode the time series into a 2D (image-based) representation, called slices",
  "name": "context_window_size",
  "type": "int"
 },
 "density": {
  "defaultValue": 0.01,
  "description": "Density of the ESN cell, where approx. `density` percent of elements being non-zero",
  "name": "density",
  "type": "float"
 },
 "imed_loss": {
  "defaultValue": False,
  "description": "Calculate loss on spatially aware (image-based) data representation instead of flat arrays",
  "name": "imed_loss",
  "type": "boolean"
 },
 "input_map_scale": {
  "defaultValue": 0.125,
  "description": "Feature scaling of the random weight preprocessing.",
  "name": "input_map_scale",
  "type": "float"
 },
 "input_map_size": {
  "defaultValue": 100,
  "description": "Size of the random weight preprocessing latent space. `input_map_size` must be larger than or equal to `context_window_size`!",
  "name": "input_map_size",
  "type": "int"
 },
 "prediction_window_size": {
  "defaultValue": 20,
  "description": "Torsk creates the input subsequences by sliding a window of size `train_window_size + prediction_window_size + 1` over the slices with shape (context_window_size, dim). `prediction_window_size` represents the size of the ESN predictions, should be `min_anomaly_length < prediction_window_size < 10 * min_anomaly_length`",
  "name": "prediction_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "reservoir_representation": {
  "defaultValue": "sparse",
  "description": "Representation of the ESN reservoirs. `sparse` is significantly faster than `dense`",
  "name": "reservoir_representation",
  "type": "enum[sparse,dense]"
 },
 "scoring_large_window_size": {
  "defaultValue": 100,
  "description": "Size of the larger of two windows slid over the prediction errors to calculate the final anomaly scores.",
  "name": "scoring_large_window_size",
  "type": "int"
 },
 "scoring_small_window_size": {
  "defaultValue": 10,
  "description": "Size of the smaller of two windows slid over the prediction errors to calculate the final anomaly scores.",
  "name": "scoring_small_window_size",
  "type": "int"
 },
 "spectral_radius": {
  "defaultValue": 2.0,
  "description": "ESN hyperparameter that determines the influence of previous internal ESN state on the next one. `spectral_radius > 1.0` increases non-linearity, but decreases short-term-memory capacity (maximized at 1.0)",
  "name": "spectral_radius",
  "type": "float"
 },
 "tikhonov_beta": {
  "defaultValue": None,
  "description": "Parameter of the Tikhonov regularization term when `train_method = tikhonov` is used.",
  "name": "tikhonov_beta",
  "type": "float"
 },
 "train_method": {
  "defaultValue": "pinv_svd",
  "description": "Solver used to train the ESN. `tikhonov` - linear solver with tikhonov regularization, `pinv_lstsq` - exact least-squares-solver that may lead to a numerical blowup, `pinv_svd` - SVD-based least-squares-solver that is highly numerically stable, but approximate",
  "name": "train_method",
  "type": "enum[pinv_lstsq,pinv_svd,tikhonov]"
 },
 "train_window_size": {
  "defaultValue": 50,
  "description": "Torsk creates the input subsequences by sliding a window of size `train_window_size + prediction_window_size + 1` over the slices with shape (context_window_size, dim). `train_window_size` represents the size of the input windows for training and prediction",
  "name": "train_window_size",
  "type": "int"
 },
 "transient_window_size": {
  "defaultValue": 10,
  "description": "Just a part of the training window, the first `transient_window_size` slices, are used for the ESN optimization.",
  "name": "transient_window_size",
  "type": "int"
 },
 "verbose": {
  "defaultValue": 2,
  "description": "Controls the logging output",
  "name": "verbose",
  "type": "int"
 }
}


def torsk(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Torsk",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/torsk",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=_post_torsk,
        param_schema=_torsk_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
