from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_lstm_vae_parameters: Dict[str, Dict[str, Any]] = {
 "batch_size": {
  "defaultValue": 32,
  "description": "size of batch given for each iteration",
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
  "defaultValue": 10,
  "description": "number of iterations we train the model",
  "name": "epochs",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 5,
  "description": "dimension of latent space",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": 0.001,
  "description": "rate at which the gradients are updated",
  "name": "learning_rate",
  "type": "float"
 },
 "lstm_layers": {
  "defaultValue": 10,
  "description": "number of layers in lstm",
  "name": "lstm_layers",
  "type": "int"
 },
 "rnn_hidden_size": {
  "defaultValue": 5,
  "description": "LTSM cells hidden dimension",
  "name": "rnn_hidden_size",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 10,
  "description": "number of datapoints that the model takes once",
  "name": "window_size",
  "type": "int"
 }
}


def lstm_vae(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LSTM-VAE",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/lstm_vae",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_lstm_vae_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
