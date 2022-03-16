from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_dbstream_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 0.1,
  "description": "For shared density: The minimum proportion of shared points between to clus-ters to warrant combining them (a suitable value for 2D data is .3). For reacha-bility clustering it is a distance factor",
  "name": "alpha",
  "type": "float"
 },
 "distance_metric": {
  "defaultValue": "Euclidean",
  "description": "The metric used to calculate distances. If shared_density is TRUE this has to be Euclidian.",
  "name": "distance_metric",
  "type": "enum[Euclidean,Manhattan,Maximum]"
 },
 "lambda": {
  "defaultValue": 0.001,
  "description": "The lambda used in the fading function.",
  "name": "lambda",
  "type": "float"
 },
 "min_weight": {
  "defaultValue": 0.0,
  "description": "The proportion of the total weight a macro-cluster needs to have not to be noise(between 0 and 1).",
  "name": "min_weight",
  "type": "float"
 },
 "n_clusters": {
  "defaultValue": 0,
  "description": "The number of macro clusters to be returned if macro is True.",
  "name": "n_clusters",
  "type": "int"
 },
 "radius": {
  "defaultValue": 0.1,
  "description": "The radius of micro-clusters.",
  "name": "radius",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "shared_density": {
  "defaultValue": "True",
  "description": "Record shared density information. If set to TRUE then shared density is used for reclustering, otherwise reachability is used (overlapping clusters with less than r\u2217(1\u2212alpha) distance are clustered together)",
  "name": "shared_density",
  "type": "boolean"
 },
 "window_size": {
  "defaultValue": 20,
  "description": "The length of the subsequences the dataset should be splitted in.",
  "name": "window_size",
  "type": "int"
 }
}


def dbstream(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DBStream",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/dbstream",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_dbstream_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
