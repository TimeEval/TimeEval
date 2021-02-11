import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, hstack
from sklearn.model_selection import ParameterGrid

from timeeval import Algorithm, AlgorithmParameter
from timeeval.adapters import DockerAdapter
from .common import SKIP_PULL


def _post_grammarviz(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
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


def grammarviz(params=None, skip_pull: bool = SKIP_PULL) -> Algorithm:
    return Algorithm(
        name="GrammarViz-docker",
        main=DockerAdapter(image_name="mut:5000/akita/grammarviz3", skip_pull=skip_pull),
        postprocess=_post_grammarviz,
        param_grid=ParameterGrid(params or {}),
        data_as_file=True
    )
