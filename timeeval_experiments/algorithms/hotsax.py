from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig

import numpy as np


import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix, hstack

from timeeval.utils.window import ReverseWindowing
from timeeval import AlgorithmParameter

# post-processing for HOT-SAX
def post_hotsax(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 100)
    if isinstance(algorithm_parameter, np.ndarray):
        results = pd.DataFrame(algorithm_parameter)
    else:
        results = pd.read_csv(algorithm_parameter)
    results.columns = ["score"]
    anomalies = results[results["score"] > .0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        tmp = np.zeros(len(results))
        tmp[idx:idx + window_size] = np.repeat([row["score"]], repeats=window_size)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx:idx + window_size] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1


_hotsax_parameters: Dict[str, Dict[str, Any]] = {
 "alphabet_size": {
  "defaultValue": 3,
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
 "normalization_threshold": {
  "defaultValue": 0.01,
  "description": "Threshold for Z-normalization of subsequences (windows). If variance of a window is higher than this threshold, it is normalized.",
  "name": "normalization_threshold",
  "type": "float"
 },
 "num_discords": {
  "defaultValue": None,
  "description": "The number of anomalies (discords) to search for in the time series. If not set, the scores for all discords are searched.",
  "name": "num_discords",
  "type": "int"
 },
 "paa_transform_size": {
  "defaultValue": 3,
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


def hotsax(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="HOT SAX",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/hotsax",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_hotsax,
        param_schema=_hotsax_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
