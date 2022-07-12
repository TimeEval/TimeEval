#!/usr/bin/env python3
from pathlib import Path

from timeeval import TimeEval, DatasetManager, DefaultMetrics, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter, FunctionAdapter
from timeeval.params import FixedParameters
from timeeval.data_types import AlgorithmParameter
import numpy as np


def your_algorithm_function(data: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return np.zeros_like(data)
    else:  # data = pathlib.Path
        return np.genfromtxt(data)[0]


def main():
    dm = DatasetManager(Path("tests/example_data"))  # or test-cases directory
    datasets = dm.select()

    algorithms = [
        Algorithm(
            name="COF",
            main=DockerAdapter(image_name="registry.gitlab.hpi.de/akita/i/cof", skip_pull=True),
            param_config=FixedParameters({
                "n_neighbors": 20,
                "random_state": 42
            }),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality("multivariate")
        ),
        Algorithm(
            name="MyPythonFunctionAlgorithm",
            main=FunctionAdapter(your_algorithm_function),
            data_as_file=False
        )
    ]

    timeeval = TimeEval(dm, datasets, algorithms,
                        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.RANGE_PR_AUC])

    timeeval.run()
    results = timeeval.get_results(aggregated=False)
    print(results)


if __name__ == "__main__":
    main()
