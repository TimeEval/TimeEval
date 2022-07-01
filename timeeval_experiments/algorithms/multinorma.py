from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_multinorma_parameters: Dict[str, Dict[str, Any]] = {
 "anomaly_window_size": {
  "defaultValue": 20,
  "description": "Sliding window size used to create subsequences (equal to desired anomaly length)",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "join_combine_method": {
  "defaultValue": 1,
  "description": "how to combine the join values from all dimensions.[0=sum, 1=max, 2=score dims (based on std, mean, range), 3=weight higher vals, 4=vals**channels]",
  "name": "join_combine_method",
  "type": "int"
 },
 "max_motifs": {
  "defaultValue": 4096,
  "description": "Maximum number of used motifs. Important to avoid OOM errors.",
  "name": "max_motifs",
  "type": "int"
 },
 "motif_detection": {
  "defaultValue": "mixed",
  "description": "Algorithm to use for motif detection [random, stomp, mixed].",
  "name": "motif_detection",
  "type": "Enum[stomp,random,mixed]"
 },
 "normal_model_percentage": {
  "defaultValue": 0.5,
  "description": "Percentage of (random) subsequences used to build the normal model.",
  "name": "normal_model_percentage",
  "type": "float"
 },
 "normalize_join": {
  "defaultValue": True,
  "description": "Apply join normalization heuristic. [False = no normalization, True = normalize]",
  "name": "normalize_join",
  "type": "boolean"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "sum_dims": {
  "defaultValue": False,
  "description": "Sum all dimensions up before computing dists, otherwise each dim is handled seperately.",
  "name": "sum_dims",
  "type": "boolean"
 }
}


def multinorma(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="MultiNormA",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/multinorma",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_multinorma_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
