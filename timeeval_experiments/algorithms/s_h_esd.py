from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_s_h_esd_parameters: Dict[str, Dict[str, Any]] = {
 "max_anomalies": {
  "defaultValue": 0.05,
  "description": "expected maximum relative frequency of anomalies in the dataset",
  "name": "max_anomalies",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "timestamp_unit": {
  "defaultValue": "m",
  "description": "If the index column ('timestamp') is of type integer, this gives the unit for date conversion. A unit less than seconds is not supported by S-H-ESD!",
  "name": "timestamp_unit",
  "type": "enum[m,h,d]"
 }
}


def s_h_esd(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="S-H-ESD (Twitter)",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/s_h_esd",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_s_h_esd_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
