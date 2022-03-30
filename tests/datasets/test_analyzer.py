import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import List, TypeVar, Dict
from unittest.mock import patch

import numpy as np

from tests.fixtures.dataset_fixtures import dataset_metadata, dataset_metadata_dict
from timeeval.datasets import DatasetAnalyzer, Stationarity, DatasetMetadata


T = TypeVar("T", DatasetMetadata, dict)


def _round_meta(metadata: T, decimals: int = 6) -> T:
    meta: T = deepcopy(metadata)

    def round_dict(dd: Dict[str, float]):
        return dict((k, np.around(v, decimals)) for k, v in dd.items())

    if isinstance(meta, DatasetMetadata):
        meta.contamination = np.around(metadata.contamination, decimals)  # type: ignore
        meta.means = round_dict(metadata.means)  # type: ignore
        meta.stddevs = round_dict(metadata.stddevs)  # type: ignore
    else:
        meta["contamination"] = np.around(metadata["contamination"], decimals)
        meta["means"] = round_dict(metadata["means"])
        meta["stddevs"] = round_dict(metadata["stddevs"])

    return meta


class TestDatasetAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = Path("tests/example_data/dataset.train.csv")

    def test_wrong_arguments(self):
        with self.assertRaises(ValueError) as e:
            DatasetAnalyzer(("test", "dataset1"), is_train=False)
        self.assertIn("must be supplied", str(e.exception))

    def test_analyzer_result(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        self.assertEqual(_round_meta(analyzer.metadata), dataset_metadata)

    def test_analyzer_result_ignore_stationarity(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path,
                                   ignore_stationarity=True)
        metadata = deepcopy(dataset_metadata)
        metadata.stationarities = {
            "value": Stationarity.NOT_STATIONARY
        }
        self.assertEqual(_round_meta(analyzer.metadata), metadata)

    def test_analyzer_result_ignore_trend(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path,
                                   ignore_trend=True)
        metadata = deepcopy(dataset_metadata)
        metadata.trends = {}
        self.assertEqual(_round_meta(analyzer.metadata), metadata)

    @patch("timeeval.datasets.analyzer.kpss")
    @patch("timeeval.datasets.analyzer.adfuller")
    def test_handles_stationaritytest_exceptions(self, adfuller_mock, kpss_mock):
        adfuller_mock.side_effect = ValueError("expected test exception")
        kpss_mock.side_effect = ValueError("expected test exception")
        metadata = deepcopy(dataset_metadata)
        metadata.stationarities = {
            "value": Stationarity.NOT_STATIONARY
        }
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        self.assertEqual(_round_meta(analyzer.metadata), metadata)

    def test_write_to_json(self):
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path)
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            analyzer.save_to_json(tmp_path / "dataset1.metadata.json")
            with open(tmp_path / "dataset1.metadata.json", "r") as f:
                meta_json = json.load(f)
        self.assertIsInstance(meta_json, List)
        self.assertDictEqual(_round_meta(meta_json[0]), dataset_metadata_dict)

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
        self.assertDictEqual(_round_meta(meta_json[1]), dataset_metadata_dict)

    def test_write_to_json_existing_overwrite(self):
        existing_entry = {"existing content untouched": False}
        analyzer = DatasetAnalyzer(("test", "dataset1"), is_train=False, dataset_path=self.dataset_path, )
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path) / "dataset1.metadata.json"
            with open(tmp_path, "w") as f:
                json.dump([existing_entry], f)
            analyzer.save_to_json(tmp_path, overwrite=True)
            with open(tmp_path, "r") as f:
                meta_json = json.load(f)
        self.assertIsInstance(meta_json, List)
        self.assertEqual(len(meta_json), 1)
        self.assertDictEqual(_round_meta(meta_json[0]), dataset_metadata_dict)
        self.assertNotIn("existing content untouched", meta_json[0])

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
        self.assertDictEqual(_round_meta(meta_json[0]), dataset_metadata_dict)
        self.assertDictEqual(meta_json_bak, existing_entry)
