from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_hbos_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 0.1,
  "description": "Regulizing alpha to prevent overflows.",
  "name": "alpha",
  "type": "float"
 },
 "bin_tol": {
  "defaultValue": 0.5,
  "description": "Parameter to decide the flexibility while dealing with the samples falling outside the bins.",
  "name": "bin_tol",
  "type": "float"
 },
 "n_bins": {
  "defaultValue": 10,
  "description": "The number of bins.",
  "name": "n_bins",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def hbos(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="HBOS",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/hbos",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_hbos_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
