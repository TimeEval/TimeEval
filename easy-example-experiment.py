#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

import numpy as np

from timeeval import TimeEval, DatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter, FunctionAdapter
from timeeval.algorithms import cof
from timeeval.data_types import AlgorithmParameter
from timeeval.params import FixedParameters


def your_algorithm_function(data: AlgorithmParameter, args: Dict[str, Any]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.zeros_like(data)
    else:  # isinstance(data, pathlib.Path)
        return np.genfromtxt(data, delimiter=",", skip_header=1)[:, 1]


def main():
    dm = DatasetManager(Path("tests/example_data"), create_if_missing=False)
    datasets = dm.select()

    algorithms = [
        Algorithm(
            name="COF",
            main=DockerAdapter(image_name="ghcr.io/timeeval/cof", skip_pull=True),
            param_config=FixedParameters({
                "n_neighbors": 20,
                "random_state": 42
            }),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE
        ),
        # or:
        # cof(params=FixedParameters({"n_neighbors": 20, "random_state": 42})),
        Algorithm(
            name="MyPythonFunctionAlgorithm",
            main=FunctionAdapter(your_algorithm_function),
            data_as_file=False
        )
    ]

    timeeval = TimeEval(dm, datasets, algorithms, metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.RANGE_PR_AUC])
    timeeval.run()
    results = timeeval.get_results(aggregated=False)
    print(results)


if __name__ == "__main__":
    main()
