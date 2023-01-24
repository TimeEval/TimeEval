from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_fast_mcd_parameters: Dict[str, Dict[str, Any]] = {
 "random_state": {
  "defaultValue": 42,
  "description": "Determines the pseudo random number generator for shuffling the data.",
  "name": "random_state",
  "type": "int"
 },
 "store_precision": {
  "defaultValue": True,
  "description": "Specify if the estimated precision is stored",
  "name": "store_precision",
  "type": "boolean"
 },
 "support_fraction": {
  "defaultValue": None,
  "description": "The proportion of points to be included in the support of the raw MCD estimate. Default is None, which implies that the minimum value of support_fraction will be used within the algorithm: `(n_sample + n_features + 1) / 2`. The parameter must be in the range (0, 1).",
  "name": "support_fraction",
  "type": "float"
 }
}


def fast_mcd(params: Optional[ParameterConfig] = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Fast-MCD",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/fast_mcd",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_fast_mcd_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
