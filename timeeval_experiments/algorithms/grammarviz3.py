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

# post-processing for GrammarViz
def post_grammarviz(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(algorithm_parameter, np.ndarray):
        results = pd.DataFrame(algorithm_parameter, columns=["index", "score", "length"])
        results = results.set_index("index")
    else:
        results = pd.read_csv(algorithm_parameter, header=None, index_col=0, names=["index", "score", "length"])
    anomalies = results[results["score"] > .0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        length = int(row["length"])
        tmp = np.zeros(len(results))
        tmp[idx:idx + length] = np.repeat([row["score"]], repeats=length)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx:idx + length] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1


_grammarviz3_parameters: Dict[str, Dict[str, Any]] = {
 "alphabet_size": {
  "defaultValue": 4,
  "description": "Number of symbols used for discretization by SAX (paper uses `\\alpha`) (performance parameter)",
  "name": "alphabet_size",
  "type": "int"
 },
 "anomaly_window_size": {
  "defaultValue": 170,
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
 "paa_transform_size": {
  "defaultValue": 4,
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


def grammarviz3(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="GrammarViz",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/grammarviz3",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_grammarviz,
        param_schema=_grammarviz3_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
