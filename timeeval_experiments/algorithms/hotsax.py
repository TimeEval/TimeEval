import numpy as np
import pandas as pd
from durations import Duration
from scipy.sparse import csc_matrix, hstack
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm, AlgorithmParameter
from timeeval.adapters import DockerAdapter
from .common import SKIP_PULL, DEFAULT_TIMEOUT


def _post_hotsax(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 20)
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


def hotsax(params=None, skip_pull: bool = SKIP_PULL, timeout: Duration = DEFAULT_TIMEOUT) -> Algorithm:
    return Algorithm(
        name="HOT-SAX-docker",
        main=DockerAdapter(image_name="mut:5000/akita/hotsax", skip_pull=skip_pull, timeout=timeout),
        postprocess=_post_hotsax,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
