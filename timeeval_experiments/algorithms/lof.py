from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_lof_parameters: Dict[str, Dict[str, Any]] = {
 "distance_metric_order": {
  "defaultValue": 2,
  "description": "Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used. See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.",
  "name": "distance_metric_order",
  "type": "int"
 },
 "leaf_size": {
  "defaultValue": 30,
  "description": "Leaf size passed to `BallTree` or `KDTree`. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
  "name": "leaf_size",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of parallel jobs to run for neighbors search. If ``-1``, then the number of jobs is set to the number of CPU cores. Affects only kneighbors and kneighbors_graph methods.",
  "name": "n_jobs",
  "type": "int"
 },
 "n_neighbors": {
  "defaultValue": 20,
  "description": "Number of neighbors to use by default for `kneighbors` queries. If n_neighbors is larger than the number of samples provided, all samples will be used.",
  "name": "n_neighbors",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def lof(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="LOF",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/lof",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_lof_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
