import json
import unittest
from copy import copy, deepcopy

import numpy as np

from tests.fixtures.dataset_fixtures import dataset_metadata
from timeeval.datasets.metadata import Trend, TrendType, DatasetMetadata, AnomalyLength, Stationarity, \
    DatasetMetadataEncoder


class TestTrend(unittest.TestCase):
    def setUp(self) -> None:
        self.linear = Trend(
            tpe=TrendType.LINEAR,
            coef=0.78,
            confidence_r2=0.7
        )
        self.quadratic = Trend(
            tpe=TrendType.QUADRATIC,
            coef=2.8,
            confidence_r2=0.58
        )
        self.kubic = Trend(
            tpe=TrendType.KUBIC,
            coef=0.973,
            confidence_r2=0.6
        )

    def test_correct_name(self):
        self.assertEqual(self.linear.name, "linear trend")
        self.assertEqual(self.quadratic.name, "quadratic trend")
        self.assertEqual(self.kubic.name, "kubic trend")

    def test_correct_order(self):
        self.assertEqual(self.linear.order, 1)
        self.assertEqual(self.quadratic.order, 2)
        self.assertEqual(self.kubic.order, 3)

    def test_trend_type_from_order(self):
        self.assertEqual(self.linear.tpe, TrendType.from_order(1))
        self.assertEqual(self.quadratic.tpe, TrendType.from_order(2))
        self.assertEqual(self.kubic.tpe, TrendType.from_order(3))


class TestDatasetMetadata(unittest.TestCase):

    def setUp(self) -> None:
        self.base = DatasetMetadata(
            dataset_id=("test", "dataset1"),
            is_train=False,
            length=300,
            dimensions=2,
            contamination=0.01,
            num_anomalies=4,
            anomaly_length=AnomalyLength(min=7, median=10, max=12),
            means={},
            stddevs={},
            trends={},
            stationarities={},
        )

    def test_mean(self):
        meta = copy(self.base)
        meta.means = {
            "value1": 2.5,
            "value2": 4.0
        }
        self.assertEqual(self.base.mean, 0)
        self.assertEqual(meta.mean, np.mean([2.5, 4.0]))

    def test_stddev(self):
        meta = copy(self.base)
        meta.stddevs = {
            "value1": 1.45,
            "value2": 0.3
        }
        self.assertEqual(self.base.stddev, 0)
        self.assertEqual(meta.stddev, np.mean([1.45, 0.3]))

    def test_trend(self):
        meta = copy(self.base)
        meta.trends = {
            "value1": [],
            "value2": [Trend(TrendType.LINEAR, 1.78, 0.67), Trend(TrendType.QUADRATIC, 0.09, 0.54)],
            "value3": [Trend(TrendType.LINEAR, 5.4, 0.9)]
        }
        self.assertEqual(self.base.trend, "no trend")
        self.assertEqual(meta.trend, "quadratic trend")

    def test_stationarity(self):
        meta = copy(self.base)
        meta.stationarities = {
            "value1": Stationarity.TREND_STATIONARY,
            "value2": Stationarity.STATIONARY,
            "value3": Stationarity.DIFFERENCE_STATIONARY
        }
        self.assertEqual(self.base.stationarity, Stationarity.STATIONARY)
        self.assertEqual(self.base.get_stationarity_name(), "stationary")
        self.assertEqual(meta.stationarity, Stationarity.TREND_STATIONARY)
        self.assertEqual(meta.get_stationarity_name(), "trend_stationary")

    def test_de_serialization(self):
        orig: DatasetMetadata = deepcopy(dataset_metadata)
        orig.trends = {
            "value1": [Trend(TrendType.LINEAR, 1.78, 0.67)]
        }
        serialized = json.dumps(orig, cls=DatasetMetadataEncoder)
        obj = json.loads(serialized, object_hook=DatasetMetadataEncoder.object_hook)
        self.assertEqual(obj, orig)

    def test_to_from_json(self):
        serialized = dataset_metadata.to_json()
        obj = DatasetMetadata.from_json(serialized)
        self.assertEqual(obj, dataset_metadata)
