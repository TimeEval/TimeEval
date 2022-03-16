from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_sarima_parameters: Dict[str, Dict[str, Any]] = {
 "exhaustive_search": {
  "defaultValue": "False",
  "description": "Performs full grid search to find optimal SARIMA-model without considering statistical tests on the data --> SLOW! but finds the optimal model.",
  "name": "exhaustive_search",
  "type": "boolean"
 },
 "max_iter": {
  "defaultValue": 20,
  "description": "The maximum number of function evaluations. smaller = faster, but might not converge.",
  "name": "max_iter",
  "type": "int"
 },
 "max_lag": {
  "defaultValue": None,
  "description": "Refit SARIMA model after that number of points (only helpful if fixed_orders=None)",
  "name": "max_lag",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of parallel jobs to run for grid search. If ``-1``, then the number of jobs is set to the number of CPU cores.",
  "name": "n_jobs",
  "type": "int"
 },
 "period": {
  "defaultValue": 1,
  "description": "Periodicity (number of periods in season), often it is 4 for quarterly data or 12 for monthly data. Default is no seasonal effect (==1). Must be >= 1.",
  "name": "period",
  "type": "int"
 },
 "prediction_window_size": {
  "defaultValue": 10,
  "description": "Number of points to forecast in one go; smaller = slower, but more accurate.",
  "name": "prediction_window_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "train_window_size": {
  "defaultValue": 500,
  "description": "Number of points from the beginning of the series to build model on.",
  "name": "train_window_size",
  "type": "int"
 }
}


def sarima(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="SARIMA",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/sarima",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_sarima_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
