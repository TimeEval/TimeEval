from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_grammarviz3_multi_parameters: Dict[str, Dict[str, Any]] = {
 "alphabet_size": {
  "defaultValue": 6,
  "description": "Number of symbols used for discretization by SAX (paper uses `\\alpha`) (performance parameter)",
  "name": "alphabet_size",
  "type": "int"
 },
 "anomaly_window_size": {
  "defaultValue": 100,
  "description": "Size of the sliding window. Equal to the discord length!",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "multi_strategy": {
  "defaultValue": 1,
  "description": "Strategy to handle multivariate output [Merge all, Merge clustered, All separate]",
  "name": "multi_strategy",
  "type": "int"
 },
 "n_discords": {
  "defaultValue": 10,
  "description": "Number of discords to report when using discord output strategy",
  "name": "n_discords",
  "type": "int"
 },
 "normalization_threshold": {
  "defaultValue": 0.01,
  "description": "Threshold for Z-normalization of subsequences (windows). If variance of a window is higher than this threshold, it is normalized.",
  "name": "normalization_threshold",
  "type": "float"
 },
 "numerosity_reduction": {
  "defaultValue": True,
  "description": "Disables / enables numerosity reduction strategy",
  "name": "numerosity_reduction",
  "type": "boolean"
 },
 "output_mode": {
  "defaultValue": 2,
  "description": "Algorithm to use for output [Density, Discord, Full]",
  "name": "output_mode",
  "type": "int"
 },
 "paa_transform_size": {
  "defaultValue": 5,
  "description": "Size of the embedding space used by PAA (paper calls it number of frames or SAX word size `w`) (performance parameter)",
  "name": "paa_transform_size",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for the random number generator",
  "name": "random_state",
  "type": "int"
 }
}


def grammarviz3_multi(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="GrammarViz-Multivariate",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/grammarviz3_multi",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_grammarviz3_multi_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
