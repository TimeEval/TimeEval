import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from timeeval.datasets.custom import CustomDatasets

DATASET_PATH = Path("tests/example_data/datasets.json")


class TestCustomDatasets(unittest.TestCase):
    @patch("timeeval.datasets.custom.Path.open")
    def test_early_conf_validation(self, mock_file):
        mock_file.return_value = StringIO('{"dataset.1": {}}')
        with self.assertRaises(ValueError) as cm:
            CustomDatasets("dummy")
            assert "must have 'dataset' and 'train_type' paths" in str(cm.exception)

    def test_get_collection_names(self):
        cd = CustomDatasets(DATASET_PATH)
        collections = cd.get_collection_names()
        assert collections == ["custom"]

    def test_get_dataset_names(self):
        cd = CustomDatasets(DATASET_PATH)
        datasets = cd.get_dataset_names()
        assert len(datasets) == 4
        assert "dataset.1" in datasets
        assert "dataset.1.train" in datasets
        assert "dataset.3" in datasets
        assert "dataset.4" in datasets

    def test_get_path(self):
        cd = CustomDatasets(DATASET_PATH)
        path = cd.get_path("dataset.1", train=False)
        assert path == Path("./tests/example_data/dataset.train.csv")

    def test_get_path_train(self):
        cd = CustomDatasets(DATASET_PATH)
        path = cd.get_path("dataset.1.train", train=True)
        assert path == Path("./tests/example_data/dataset.train.csv")

    def test_get_path_wrong_type(self):
        cd = CustomDatasets(DATASET_PATH)
        with self.assertRaises(ValueError) as cm:
            cd.get_path("dataset.1", train=True)
            assert "meant for testing and not for training" in str(cm.exception)
        with self.assertRaises(ValueError) as cm:
            cd.get_path("dataset.1.train", train=False)
            assert "meant for training and not for testing" in str(cm.exception)


if __name__ == "__main__":
    unittest.main()
