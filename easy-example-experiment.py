#!/usr/bin/env python3
from pathlib import Path


from timeeval import TimeEval, DatasetManager, Metric, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import FixedParameters


def main():
    dm = DatasetManager(Path("tests/example_data"))  # or test-cases directory
    datasets = dm.select()

    algorithm = Algorithm(
        name="COF",
        main=DockerAdapter(image_name="mut:5000/akita/cof", skip_pull=True),
        param_config=FixedParameters({
            "n_neighbors": 20,
            "random_state": 42
        }),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )

    timeeval = TimeEval(dm, datasets, [algorithm], metrics=[Metric.ROC_AUC, Metric.RANGE_PR_AUC])

    timeeval.run()
    print(timeeval.get_results(aggregated=False))


if __name__ == "__main__":
    main()
