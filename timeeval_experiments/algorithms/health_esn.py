from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_health_esn_parameters: Dict[str, Dict[str, Any]] = {
 "activation": {
  "defaultValue": "tanh",
  "description": "Activation function used for the ESN.",
  "name": "activation",
  "type": "enum[tanh,sigmoid]"
 },
 "connectivity": {
  "defaultValue": 0.25,
  "description": "How dense the units in the reservoir are connected (= percentage of non-zero weights)",
  "name": "connectivity",
  "type": "float"
 },
 "linear_hidden_size": {
  "defaultValue": 500,
  "description": "Hidden units in ESN reservoir.",
  "name": "linear_hidden_size",
  "type": "int"
 },
 "prediction_window_size": {
  "defaultValue": 20,
  "description": "Window of predicted points in the future.",
  "name": "prediction_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "spectral_radius": {
  "defaultValue": 0.6,
  "description": "Factor used for random initialization of ESN neural connections.",
  "name": "spectral_radius",
  "type": "float"
 }
}


def health_esn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="HealthESN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/health_esn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_health_esn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
