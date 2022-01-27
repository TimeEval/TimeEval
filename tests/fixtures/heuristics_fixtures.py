from pathlib import Path

from tests.fixtures.algorithms import DeviatingFromMean
from timeeval import Algorithm, TrainingType
from timeeval.datasets import Dataset


algorithm = Algorithm(
    name="deviating_from_mean",
    main=DeviatingFromMean(),
    training_type=TrainingType.UNSUPERVISED,
    data_as_file=False
)
dataset = Dataset(
    datasetId=("test", "dataset-datetime"),
    dataset_type="synthetic",
    training_type=TrainingType.SUPERVISED,
    num_anomalies=3,
    dimensions=1,
    length=3000,
    contamination=0.0002777777777777778,
    min_anomaly_length=1,
    median_anomaly_length=5,
    max_anomaly_length=20,
    period_size=50
)
dummy_dataset_path = Path("dummy")
real_test_dataset_path = Path("tests/example_data/dataset.test.csv")
