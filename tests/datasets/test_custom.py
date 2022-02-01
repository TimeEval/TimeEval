import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tests.fixtures.dataset_fixtures import CUSTOM_DATASET_PATH
from timeeval import TrainingType
from timeeval.datasets import Dataset
from timeeval.datasets.custom import CustomDatasets, TEST_PATH_KEY, TRAIN_PATH_KEY


class TestCustomDatasets(unittest.TestCase):
    @patch("timeeval.datasets.custom.Path.open")
    def test_early_conf_validation(self, mock_file):
        mock_file.return_value = StringIO('{"dataset.1": {}}')
        with self.assertRaises(ValueError) as cm:
            CustomDatasets("dummy")
        assert f"misses the required '{TEST_PATH_KEY}' property" in str(cm.exception)

        mock_file.return_value = StringIO('{"dataset.1": {"test_path": "missing"}}')
        with self.assertRaises(ValueError) as cm:
            CustomDatasets("dummy")
        assert f"was not found (property '{TEST_PATH_KEY}')" in str(cm.exception)

        mock_file.return_value = StringIO(
            '{"dataset.1": {"test_path": "./tests/example_data/data.txt", "train_path": "missing"}}'
        )
        with self.assertRaises(ValueError) as cm:
            CustomDatasets("dummy")
        assert f"was not found (property '{TRAIN_PATH_KEY}')" in str(cm.exception)

    def test_get_collection_names(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        collections = cd.get_collection_names()
        assert collections == ["custom"]

    def test_get_dataset_names(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        datasets = cd.get_dataset_names()
        assert len(datasets) == 4
        assert "dataset.1" in datasets
        assert "dataset.1.train" in datasets
        assert "dataset.3" in datasets
        assert "dataset.4" in datasets

    def test_get_path(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        path = cd.get_path("dataset.1.train", train=False)
        assert path == Path("./tests/example_data/dataset.test.csv").resolve()

    def test_get_path_train(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        path = cd.get_path("dataset.1.train", train=True)
        assert path == Path("./tests/example_data/dataset.train.csv").resolve()

    def test_get_path_wrong_type(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        with self.assertRaises(ValueError) as cm:
            cd.get_path("dataset.1", train=True)
        assert "unsupervised and has no training time series" in str(cm.exception)
        with self.assertRaises(ValueError) as cm:
            cd.get_path("dataset.3", train=True)
        assert "unsupervised and has no training time series" in str(cm.exception)

    def test_get(self):
        cd = CustomDatasets(CUSTOM_DATASET_PATH)
        dataset = cd.get("dataset.1")
        assert dataset == Dataset(
            datasetId=("custom", "dataset.1"),
            dataset_type="synthetic test",
            training_type=TrainingType.UNSUPERVISED,
            dimensions=1,
            length=3600,
            contamination=0.0002777777777777778,
            min_anomaly_length=1,
            median_anomaly_length=1,
            max_anomaly_length=1,
            num_anomalies=1,
            period_size=10
        )
        assert dataset.has_anomalies
