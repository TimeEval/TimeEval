from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_arima_parameters: Dict[str, Dict[str, Any]] = {
 "differencing_degree": {
  "defaultValue": 0,
  "description": "Differencing degree for the auto-ARIMA process",
  "name": "differencing_degree",
  "type": "int"
 },
 "distance_metric": {
  "defaultValue": "Euclidean",
  "description": "Distance measure used to calculate the prediction error = anomaly score",
  "name": "distance_metric",
  "type": "enum[Euclidean,Mahalanobis,Garch,SSA,Fourier,DTW,EDRS,TWED]"
 },
 "max_lag": {
  "defaultValue": 30000,
  "description": "Number of points, after which the ARIMA model is re-fitted to the data to deal with trends and shifts",
  "name": "max_lag",
  "type": "int"
 },
 "max_p": {
  "defaultValue": 5,
  "description": "Maximum AR-order for the auto-ARIMA process",
  "name": "max_p",
  "type": "int"
 },
 "max_q": {
  "defaultValue": 5,
  "description": "Maximum MA-order for the auto-ARIMA process",
  "name": "max_q",
  "type": "int"
 },
 "p_start": {
  "defaultValue": 1,
  "description": "Minimum AR-order for the auto-ARIMA process",
  "name": "p_start",
  "type": "int"
 },
 "q_start": {
  "defaultValue": 1,
  "description": "Minimum MA-order for the auto-ARIMA process",
  "name": "q_start",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "window_size": {
  "defaultValue": 20,
  "description": "Size of sliding window (also used as prediction window size)",
  "name": "window_size",
  "type": "int"
 }
}


def arima(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="ARIMA",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/arima",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_arima_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
