import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from timeeval import Datasets, DatasetManager
from timeeval.datasets import DatasetRecord, DatasetMetadata, AnomalyLength, Stationarity


CUSTOM_DATASET_PATH = Path("tests/example_data/datasets.json")
custom_dataset_names = ["dataset.1", "dataset.1.train", "dataset.3", "dataset.4"]

dataset_index_header = "collection_name,dataset_name,train_path,test_path,dataset_type,datetime_index,split_at,train_type,train_is_normal,input_type,length,dimensions,contamination,num_anomalies,min_anomaly_length,median_anomaly_length,max_anomaly_length,mean,stddev,trend,stationarity,period_size"
dataset_index_content_nab = "NAB,art_daily_no_noise,,data-processed/univariate/NAB/art_daily_no_noise.test.csv,synthetic,True,,unsupervised,False,univariate,4032,1,0.015,1,5,5,6,564.52,2.468,no trend,not_stationary,"
dataset_index_content_test = "test-collection,test_dataset,path_train.csv,path_test.csv,real,True,,supervised,True,univariate,12,1,0.01,2,5,5,6,564.52,2.468,no trend,not_stationary,"
nab_record = DatasetRecord(
    collection_name="NAB",
    dataset_name="art_daily_no_noise",
    train_path=None,
    test_path="data-processed/univariate/NAB/art_daily_no_noise.test.csv",
    dataset_type="synthetic",
    datetime_index=True,
    split_at=np.nan,  # type: ignore
    train_type="unsupervised",
    train_is_normal=False,
    input_type="univariate",
    length=4032,
    dimensions=1,
    contamination=0.015,
    num_anomalies=1,
    min_anomaly_length=5,
    median_anomaly_length=5,
    max_anomaly_length=6,
    mean=564.52,
    stddev=2.468,
    trend="no trend",
    stationarity="not_stationary",
    period_size=None
)
test_record = DatasetRecord(
    collection_name="test-collection",
    dataset_name="test_dataset",
    train_path="path_train.csv",
    test_path="path_test.csv",
    dataset_type="real",
    datetime_index=True,
    split_at=np.nan,  # type: ignore
    train_type="supervised",
    train_is_normal=True,
    input_type="univariate",
    length=12,
    dimensions=1,
    contamination=0.01,
    num_anomalies=2,
    min_anomaly_length=5,
    median_anomaly_length=5,
    max_anomaly_length=6,
    mean=564.52,
    stddev=2.468,
    trend="no trend",
    stationarity="not_stationary",
    period_size=None
)
# excerpt from NYC taxi dataset
dataset_content = """
timestamp,value,is_anomaly
2014-10-30 09:30:00,18967,0
2014-10-30 10:00:00,17899,0
2014-10-30 10:30:00,17994,0
2014-10-30 11:00:00,17167,0
2014-10-30 11:30:00,18094,0
2014-10-30 12:00:00,18575,0
2014-10-30 12:30:00,18022,0
2014-10-30 13:00:00,17359,0
2014-10-30 13:30:00,18035,0
2014-10-30 14:00:00,18733,0
2014-10-30 14:30:00,19410,0
2014-10-30 15:00:00,18991,0
2014-10-30 15:30:00,16749,1
2014-10-30 16:00:00,14604,1
2014-10-30 16:30:00,13367,1
2014-10-30 17:00:00,16382,1
"""
dataset_content_io = StringIO(dataset_content)
dataset_df = pd.read_csv(dataset_content_io, parse_dates=["timestamp"], infer_datetime_format=True)
dataset_content_io.seek(0)
dataset_ndarray = dataset_df.values


def fill_file(path, lines=(dataset_index_content_nab,)):
    with open(path / Datasets.INDEX_FILENAME, "w") as f:
        f.write(dataset_index_header)
        f.write("\n")
        for l in lines:
            f.write(l)
            f.write("\n")


def read_file(path):
    assert os.path.isfile(path / Datasets.INDEX_FILENAME)
    with open(path / Datasets.INDEX_FILENAME, "r") as f:
        return [l.strip() for l in f.readlines()]


def add_dataset(dm: DatasetManager):
    dm.add_dataset(DatasetRecord(
        collection_name=test_record.collection_name,
        dataset_name=test_record.dataset_name,
        train_path=test_record.train_path,
        test_path=test_record.test_path,
        dataset_type=test_record.dataset_type,
        datetime_index=test_record.datetime_index,
        split_at=test_record.split_at,
        train_type=test_record.train_type,
        train_is_normal=test_record.train_is_normal,
        input_type=test_record.input_type,
        length=test_record.length,
        dimensions=test_record.dimensions,
        contamination=test_record.contamination,
        num_anomalies=test_record.num_anomalies,
        min_anomaly_length=test_record.min_anomaly_length,
        median_anomaly_length=test_record.median_anomaly_length,
        max_anomaly_length=test_record.max_anomaly_length,
        mean=test_record.mean,
        stddev=test_record.stddev,
        trend=test_record.trend,
        stationarity=test_record.stationarity,
        period_size=None
    ))


dataset_metadata = DatasetMetadata(
    dataset_id=("test", "dataset1"),
    is_train=False,
    length=3600,
    dimensions=1,
    contamination=0.000278,
    num_anomalies=1,
    anomaly_length=AnomalyLength(min=1, median=1, max=1),
    means={"value": 15836.711944},
    stddevs={"value": 7084.011043},
    trends={"value": []},
    stationarities={"value": Stationarity.DIFFERENCE_STATIONARY}
)
dataset_metadata_dict = {
    "dataset_id": ["test", "dataset1"],
    "is_train": False,
    "length": 3600,
    "dimensions": 1,
    "contamination": 0.000278,
    "num_anomalies": 1,
    "anomaly_length": {"min": 1, "median": 1, "max": 1},
    "means": {"value": 15836.711944},
    "stddevs": {"value": 7084.011043},
    "trends": {"value": []},
    "stationarities": {"value": "difference_stationary"}
}
