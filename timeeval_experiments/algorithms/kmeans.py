from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_kmeans_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Size of sliding windows. The bigger `window_size` is, the bigger the anomaly context is. If it's to big, things seem anomalous that are not. If it's too small, the algorithm is not able to find anomalous windows and looses its time context.",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "n_clusters": {
  "defaultValue": 20,
  "description": "The number of clusters to form as well as the number of centroids to generate. The bigger `n_clusters` (`k`) is, the less noisy the anomaly scores are.",
  "name": "n_clusters",
  "type": "int"
 },
 "n_jobs": {
  "defaultValue": 1,
  "description": "Internal parallelism used (sample-wise in the main loop which assigns each sample to its closest center). If `-1` or `None`, all available CPUs are used.",
  "name": "n_jobs",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "stride": {
  "defaultValue": 1,
  "description": "Stride of sliding windows. It is the step size between windows. The larger `stride` is, the noisier the scores get. If `stride == window_size`, they are tumbling windows.",
  "name": "stride",
  "type": "int"
 }
}


def kmeans(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="k-Means",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/kmeans",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_kmeans_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
