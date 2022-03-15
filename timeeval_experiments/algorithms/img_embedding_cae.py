from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_img_embedding_cae_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 512,
  "description": "length of one time series chunk (tumbling window)",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "batch_size": {
  "defaultValue": 32,
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
  "defaultValue": 30,
  "description": "number of training iterations over entire dataset",
  "name": "epochs",
  "type": "int"
 },
 "kernel_size": {
  "defaultValue": 2,
  "description": "width, height of each convolution kernel (stride is equal to this value)",
  "name": "kernel_size",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 100,
  "description": "number of neurons used in the embedding layer",
  "name": "latent_size",
  "type": "int"
 },
 "leaky_relu_alpha": {
  "defaultValue": 0.03,
  "description": "alpha value used for leaky relu activation function",
  "name": "leaky_relu_alpha",
  "type": "float"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "Gradient factor for backpropagation",
  "name": "learning_rate",
  "type": "float"
 },
 "num_kernels": {
  "defaultValue": 64,
  "description": "number of convolution kernels used in each layer",
  "name": "num_kernels",
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
  "defaultValue": 128,
  "description": "number of simultaneously trained data instances",
  "name": "test_batch_size",
  "type": "int"
 }
}


def img_embedding_cae(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="ImageEmbeddingCAE",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/img_embedding_cae",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_img_embedding_cae_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
