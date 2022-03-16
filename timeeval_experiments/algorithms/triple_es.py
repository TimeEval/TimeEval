from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_triple_es_parameters: Dict[str, Dict[str, Any]] = {
 "period": {
  "defaultValue": 100,
  "description": "number of time units at which events happen regularly/periodically",
  "name": "period",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "seasonal": {
  "defaultValue": "add",
  "description": "type of seasonal component",
  "name": "seasonal",
  "type": "enum[add, mul]"
 },
 "train_window_size": {
  "defaultValue": 200,
  "description": "size of each TripleES model to predict the next timestep",
  "name": "train_window_size",
  "type": "int"
 },
 "trend": {
  "defaultValue": "add",
  "description": "type of trend component",
  "name": "trend",
  "type": "enum[add, mul]"
 }
}


def triple_es(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Triple ES (Holt-Winter's)",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/triple_es",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_triple_es_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
