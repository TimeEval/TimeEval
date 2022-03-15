from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_cblof_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 0.9,
  "description": "Coefficient for deciding small and large clusters. The ratio of the number of samples in large clusters to the number of samples in small clusters. (0.5 < alpha < 1)",
  "name": "alpha",
  "type": "float"
 },
 "beta": {
  "defaultValue": 5,
  "description": "Coefficient for deciding small and large clusters. For a list sorted clusters by size `|C1|, |C2|, ..., |Cn|, beta = |Ck|/|Ck-1|`. (1.0 < beta )",
  "name": "beta",
  "type": "float"
 },
 "n_clusters": {
  "defaultValue": 8,
  "description": "The number of clusters to form as well as the number of centroids to generate.",
  "name": "n_clusters",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of parallel jobs to run for neighbors search. If `-1`, then the number of jobs is set to the number of CPU cores. Affects only kneighbors and kneighbors_graph methods.",
  "name": "n_jobs",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "use_weights": {
  "defaultValue": "False",
  "description": "If set to True, the size of clusters are used as weights in outlier score calculation.",
  "name": "use_weights",
  "type": "boolean"
 }
}


def cblof(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="CBLOF",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/cblof",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_cblof_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
