import json
import tempfile
import unittest
from pathlib import Path
from typing import List

from tests.fixtures.dataset_fixtures import dataset_metadata, dataset_metadata_dict
from timeeval.datasets import DatasetAnalyzer


class TestDatasetAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = Path("tests/example_data/dataset.train.csv")

    def test_wrong_arguments(self):
        with self.assertRaises(ValueError) as e:
            DatasetAnalyzer(("test", "dataset1"), is_train=False)
            self.assertIn("must be supplied", str(e))

    def test_analyzer_result(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        self.assertEqual(analyzer.metadata, dataset_metadata)

    def test_write_to_json(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            analyzer.save_to_json(tmp_path / "dataset1.metadata.json")
            with open(tmp_path / "dataset1.metadata.json", "r") as f:
                meta_json = json.load(f)
        self.assertIsInstance(meta_json, List)
        self.assertDictEqual(meta_json[0], dataset_metadata_dict)

    def test_write_to_json_existing(self):
        existing_entry = {"existing content untouched": True}
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "dataset1.metadata.json"
            with open(tmp_path, "w") as f:
                json.dump([existing_entry], f)
            analyzer.save_to_json(tmp_path)
            with open(tmp_path, "r") as f:
                meta_json = json.load(f)
        self.assertIsInstance(meta_json, List)
        self.assertDictEqual(meta_json[0], existing_entry)
        self.assertDictEqual(meta_json[1], dataset_metadata_dict)

    def test_write_to_json_existing_broken(self):
        existing_entry = {"wrongly formatted": "existing json"}
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "dataset1.metadata.json"
            with open(tmp_path, "w") as f:
                json.dump(existing_entry, f)
            analyzer.save_to_json(tmp_path)
            with open(tmp_path, "r") as f:
                meta_json = json.load(f)
            with open(tmp_path.parent / "dataset1.metadata.json.bak", "r") as f:
                meta_json_bak = json.load(f)
        self.assertIsInstance(meta_json, List)
        self.assertDictEqual(meta_json[0], dataset_metadata_dict)
        self.assertDictEqual(meta_json_bak, existing_entry)
