from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_autoencoder_parameters: Dict[str, Dict[str, Any]] = {
 "early_stopping_delta": {
  "defaultValue": 0.5,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_delta",
  "type": "float"
 },
 "early_stopping_patience": {
  "defaultValue": 10,
  "description": "If loss is `delta` or less smaller for `patience` epochs, stop",
  "name": "early_stopping_patience",
  "type": "int"
 },
 "epochs": {
  "defaultValue": 10,
  "description": "Number of training epochs",
  "name": "epochs",
  "type": "int"
 },
 "latent_size": {
  "defaultValue": 32,
  "description": "Dimensionality of the latent space",
  "name": "latent_size",
  "type": "int"
 },
 "learning_rate": {
  "defaultValue": "0.005",
  "description": "Learning rate",
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
  "description": "Fraction to split training data by for validation",
  "name": "split",
  "type": "float"
 }
}


def autoencoder(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="AutoEncoder (AE)",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/autoencoder",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_autoencoder_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
