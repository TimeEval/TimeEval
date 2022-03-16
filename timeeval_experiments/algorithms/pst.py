from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_pst_parameters: Dict[str, Dict[str, Any]] = {
 "max_depth": {
  "defaultValue": 4,
  "description": "Maximal depth of the PST. Default to maximum length of the sequence(s) in object minus 1.",
  "name": "max_depth",
  "type": "int"
 },
 "n_bins": {
  "defaultValue": 5,
  "description": "Number of Bags (bins) in which the time-series should be splitted by frequency.",
  "name": "n_bins",
  "type": "int"
 },
 "n_min": {
  "defaultValue": 1,
  "description": "Minimum number of occurences of a string to add it in the tree.",
  "name": "n_min",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "sim": {
  "defaultValue": "SIMn",
  "description": "The similarity measure to use when computing the similarity between a sequence and the pst. SIMn is supposed to yield better results.",
  "name": "sim",
  "type": "enum[SIMo,SIMn]"
 },
 "window_size": {
  "defaultValue": 5,
  "description": "Length of the subsequences in which the time series should be splitted into (sliding window).",
  "name": "window_size",
  "type": "int"
 },
 "y_min": {
  "defaultValue": None,
  "description": "Smoothing parameter for conditional probabilities, assuring that nosymbol, and hence no sequence, is predicted to have a None probability. The parameter $ymin$ sets a lower bound for a symbol\u2019s probability.",
  "name": "y_min",
  "type": "float"
 }
}


def pst(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="PST",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/pst",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_pst_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
