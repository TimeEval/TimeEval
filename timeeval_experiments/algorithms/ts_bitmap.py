from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_ts_bitmap_parameters: Dict[str, Dict[str, Any]] = {
 "alphabet_size": {
  "defaultValue": 5,
  "description": "Number of bins for SAX discretization.",
  "name": "alphabet_size",
  "type": "int"
 },
 "compression_ratio": {
  "defaultValue": 2,
  "description": "How much to compress the timeseries in the PAA step. If `compression_ration == 1`, no compression.",
  "name": "compression_ratio",
  "type": "int"
 },
 "feature_window_size": {
  "defaultValue": 100,
  "description": "Size of the tumbling windows used for SAX discretization.",
  "name": "feature_window_size",
  "type": "int"
 },
 "lag_window_size": {
  "defaultValue": 300,
  "description": "How far to look back to create the lag bitmap.",
  "name": "lag_window_size",
  "type": "int"
 },
 "lead_window_size": {
  "defaultValue": 200,
  "description": "How far to look ahead to create lead bitmap.",
  "name": "lead_window_size",
  "type": "int"
 },
 "level_size": {
  "defaultValue": 3,
  "description": "Desired level of recursion of the bitmap.",
  "name": "level_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def ts_bitmap(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="TSBitmap",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/ts_bitmap",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_ts_bitmap_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
