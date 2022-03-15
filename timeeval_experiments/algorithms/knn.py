from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_knn_parameters: Dict[str, Dict[str, Any]] = {
 "distance_metric_order": {
  "defaultValue": 2,
  "description": "Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used. See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.",
  "name": "distance_metric_order",
  "type": "int"
 },
 "leaf_size": {
  "defaultValue": 30,
  "description": "Leaf size passed to `BallTree`. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
  "name": "leaf_size",
  "type": "int"
 },
 "method": {
  "defaultValue": "largest",
  "description": "'largest': use the distance to the kth neighbor as the outlier score, 'mean': use the average of all k neighbors as the outlier score, 'median': use the median of the distance to k neighbors as the outlier score.",
  "name": "method",
  "type": "enum[largest,mean,median]"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "The number of parallel jobs to run for neighbors search. If ``-1``, then the number of jobs is set to the number of CPU cores. Affects only kneighbors and kneighbors_graph methods.",
  "name": "n_jobs",
  "type": "int"
 },
 "n_neighbors": {
  "defaultValue": 5,
  "description": " Number of neighbors to use by default for `kneighbors` queries.",
  "name": "n_neighbors",
  "type": "int"
 },
 "radius": {
  "defaultValue": 1.0,
  "description": " Range of parameter space to use by default for `radius_neighbors` queries.",
  "name": "radius",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 }
}


def knn(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="KNN",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/knn",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_knn_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
