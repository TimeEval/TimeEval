from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


from timeeval.utils.window import ReverseWindowing
# post-processing for sLOF
def post_sLOF(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)


_subsequence_lof_multi_sum_parameters: Dict[str, Dict[str, Any]] = {
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
 },
 "window_size": {
  "defaultValue": 100,
  "description": "Size of the sliding windows to extract subsequences as input to LOF.",
  "name": "window_size",
  "type": "int"
 }
}


def subsequence_lof_multi_sum(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Subsequence LOF Multivariate Sum",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/subsequence_lof_multi_sum",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_sLOF,
        param_schema=_subsequence_lof_multi_sum_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
