from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_ensemble_gi_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 50,
  "description": "The size of the sliding window, in which `w` regions are made discrete.",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "max_alphabet_size": {
  "defaultValue": 10,
  "description": "Maximum number of symbols used for discretization by SAX (`\\alpha`)",
  "name": "max_alphabet_size",
  "type": "int"
 },
 "max_paa_transform_size": {
  "defaultValue": 20,
  "description": "Maximum size of the embedding space used by PAA (SAX word size `w`)",
  "name": "max_paa_transform_size",
  "type": "int"
 },
 "n_estimators": {
  "defaultValue": 10,
  "description": "The number of models in the ensemble.",
  "name": "n_estimators",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of parallel jobs to use for executing the models. If `-1`, then the number of jobs is set to the number of CPU cores.",
  "name": "n_jobs",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 },
 "selectivity": {
  "defaultValue": 0.8,
  "description": "The fraction of models in the ensemble included in the end result.",
  "name": "selectivity",
  "type": "float"
 },
 "window_method": {
  "defaultValue": "sliding",
  "description": "Windowing method used to create subsequences. The original implementation had a strange method (`orig`) that is similar to `tumbling`, the paper uses a `sliding` window. However, `sliding` is significantly slower than `tumbling` while producing better results (higher anomaly score resolution). `orig` should not be used!",
  "name": "window_method",
  "type": "enum[sliding,tumbling,orig]"
 }
}


def ensemble_gi(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Ensemble GI",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/ensemble_gi",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_ensemble_gi_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
