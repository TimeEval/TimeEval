from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_dspot_parameters: Dict[str, Dict[str, Any]] = {
 "alert": {
  "defaultValue": "True",
  "description": "Enable alert triggering, if False, even out-of-bounds-data will be taken into account for tail fit",
  "name": "alert",
  "type": "boolean"
 },
 "bounded": {
  "defaultValue": "True",
  "description": "Performance: enable memory bounding (also improves performance)",
  "name": "bounded",
  "type": "boolean"
 },
 "down": {
  "defaultValue": "True",
  "description": "Compute lower thresholds",
  "name": "down",
  "type": "boolean"
 },
 "level": {
  "defaultValue": 0.99,
  "description": "Calibration: proportion of initial data (n_init) not involved in the tail distribution fit during initialization. The user must ensure that n_init * (1 - level) > 10",
  "name": "level",
  "type": "float"
 },
 "max_excess": {
  "defaultValue": 200,
  "description": "Performance: maximum number of data stored to perform the tail fit when memory bounding is enabled",
  "name": "max_excess",
  "type": "int"
 },
 "n_init": {
  "defaultValue": 1000,
  "description": "Calibration: number of data used to calibrate algorithm. The user must ensure that n_init * (1 - level) > 10",
  "name": "n_init",
  "type": "int"
 },
 "q": {
  "defaultValue": 0.001,
  "description": "Main parameter: maximum probability of an abnormal event",
  "name": "q",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "up": {
  "defaultValue": "True",
  "description": "Compute upper thresholds",
  "name": "up",
  "type": "boolean"
 }
}


def dspot(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DSPOT",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/dspot",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_dspot_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
