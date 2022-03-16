from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_ocean_wnn_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 64,
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
  "defaultValue": 1,
  "description": "Number of training iterations over entire dataset; recommended value: 1000",
  "name": "epochs",
  "type": "int"
 },
 "hidden_size": {
  "defaultValue": 20,
  "description": "Number of neurons in hidden layer",
  "name": "hidden_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.01,
  "description": "Learning rate for Adam optimizer",
  "name": "learning_rate",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "split": {
  "defaultValue": 0.8,
  "description": "Train-validation split for early stopping",
  "name": "split",
  "type": "float"
 },
 "test_batch_size": {
  "defaultValue": 256,
  "description": "Batch size over test and validation dataset",
  "name": "test_batch_size",
  "type": "int"
 },
 "threshold_percentile": {
  "defaultValue": 0.99,
  "description": "Upper percentile of training residual distribution used for detection replacement.",
  "name": "threshold_percentile",
  "type": "float"
 },
 "train_window_size": {
  "defaultValue": 20,
  "description": "Window size used for forecasting the next point",
  "name": "train_window_size",
  "type": "int"
 },
 "wavelet_a": {
  "defaultValue": -2.5,
  "description": "WBF scale parameter; recommended range: [-2.5, 2.5]",
  "name": "wavelet_a",
  "type": "float"
 },
 "wavelet_cs_C": {
  "defaultValue": 1.75,
  "description": "Cosine factor for central-symmetric WBF.",
  "name": "wavelet_cs_C",
  "type": "float"
 },
 "wavelet_k": {
  "defaultValue": -1.5,
  "description": "WBF shift parameter; recommended range: [-1.5, 1.5]",
  "name": "wavelet_k",
  "type": "float"
 },
 "wavelet_wbf": {
  "defaultValue": "mexican_hat",
  "description": "Mother WBF; allowed values: \"mexican_hat\", \"central_symmetric\", \"morlet\"",
  "name": "wavelet_wbf",
  "type": "enum[mexican_hat,central_symmetric,morlet]"
 },
 "with_threshold": {
  "defaultValue": "True",
  "description": "If True, values whose forecasting error exceeds the threshold are not included in next window, but are replaced by the prediction.",
  "name": "with_threshold",
  "type": "boolean"
 }
}


def ocean_wnn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="OceanWNN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/ocean_wnn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_ocean_wnn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
