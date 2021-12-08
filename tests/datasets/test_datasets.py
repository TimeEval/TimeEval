import os
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from timeeval import Datasets, DatasetManager, InputDimensionality, TrainingType
from timeeval.datasets import DatasetRecord


header = "collection_name,dataset_name,train_path,test_path,dataset_type,datetime_index,split_at,train_type,train_is_normal,input_type,length,dimensions,contamination,num_anomalies,min_anomaly_length,median_anomaly_length,max_anomaly_length,mean,stddev,trend,stationarity,period_size"
content_nab = "NAB,art_daily_no_noise,,data-processed/univariate/NAB/art_daily_no_noise.test.csv,synthetic,True,,unsupervised,False,univariate,4032,1,0.01,2,5,5,6,564.52,2.468,no trend,not_stationary,"
content_test = "test-collection,test_dataset,path_train.csv,path_test.csv,real,True,,supervised,True,univariate,12,1,0.01,2,5,5,6,564.52,2.468,no trend,not_stationary,"
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
dataset_ndarray = np.genfromtxt(dataset_content_io, delimiter=",", skip_header=1)
custom_dataset_names = ["dataset.1", "dataset.1.train", "dataset.3", "dataset.4"]


def _fill_file(path, lines=(content_nab,)):
    with open(path / Datasets.INDEX_FILENAME, "w") as f:
        f.write(header)
        f.write("\n")
        for l in lines:
            f.write(l)
            f.write("\n")


def _read_file(path):
    assert os.path.isfile(path / Datasets.INDEX_FILENAME)
    with open(path / Datasets.INDEX_FILENAME, "r") as f:
        return [l.strip() for l in f.readlines()]


def test_initialize_empty_folder(tmp_path):
    DatasetManager(data_folder=tmp_path)
    assert os.path.isfile(tmp_path / Datasets.INDEX_FILENAME)


def test_initialize_missing_folder(tmp_path):
    target = tmp_path / "subfolder"
    DatasetManager(data_folder=target)
    assert os.path.isfile(target / Datasets.INDEX_FILENAME)


def test_initialize_raise_error_empty_folder(tmp_path):
    target = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as ex:
        DatasetManager(data_folder=target, create_if_missing=False)
        assert "Could not find the index file" in str(ex.value)


def test_initialize_existing(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def test_refresh(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == []
    assert list(dm.get_dataset_names()) == []

    with open(tmp_path / Datasets.INDEX_FILENAME, "a") as f:
        f.write(content_nab)
        f.write("\n")

    dm.refresh()
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def _test_select_helper(path, lines=(content_nab, content_test), **kwargs):
    _fill_file(path, lines)
    dm = DatasetManager(data_folder=path)
    return dm.select(**kwargs)


def test_select_collection(tmp_path):
    names = _test_select_helper(tmp_path, collection=nab_record.collection_name)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_name(tmp_path):
    names = _test_select_helper(tmp_path, dataset=nab_record.dataset_name)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_dataset_type(tmp_path):
    names = _test_select_helper(tmp_path, dataset_type=nab_record.dataset_type)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_datetime_index(tmp_path):
    names = _test_select_helper(tmp_path, datetime_index=nab_record.datetime_index)
    assert names == [(nab_record.collection_name, nab_record.dataset_name),
                     (test_record.collection_name, test_record.dataset_name)]


def test_select_train_type(tmp_path):
    names = _test_select_helper(tmp_path, training_type=TrainingType(nab_record.train_type))
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_train_is_normal(tmp_path):
    names = _test_select_helper(tmp_path, train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_input_type(tmp_path):
    names = _test_select_helper(tmp_path, input_dimensionality=InputDimensionality(nab_record.input_type))
    assert names == [(nab_record.collection_name, nab_record.dataset_name),
                     (test_record.collection_name, test_record.dataset_name)]


def test_select_combined(tmp_path):
    names = _test_select_helper(tmp_path,
                                collection=nab_record.collection_name,
                                train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_wrong_order(tmp_path):
    names = _test_select_helper(tmp_path,
                                lines=[content_test, content_nab],
                                collection=test_record.collection_name,
                                train_is_normal=test_record.train_is_normal)
    assert names == [(test_record.collection_name, test_record.dataset_name)]


def _test_select_with_custom_helper(path, **kwargs):
    _fill_file(path)
    dm = DatasetManager(data_folder=path, custom_datasets_file="tests/example_data/datasets.json")
    return dm.select(**kwargs)


def test_select_with_custom(tmp_path):
    names = _test_select_with_custom_helper(tmp_path,
                                            collection=nab_record.collection_name,
                                            train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]
    names = _test_select_with_custom_helper(tmp_path, dataset=nab_record.dataset_name)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_with_custom_dataset(tmp_path):
    names = _test_select_with_custom_helper(tmp_path, collection="custom", dataset="dataset.1")
    assert names == [("custom", "dataset.1")]
    names = _test_select_with_custom_helper(tmp_path, collection="custom")
    assert names == [("custom", name) for name in custom_dataset_names]
    names = _test_select_with_custom_helper(tmp_path, dataset="dataset.1")
    assert names == [("custom", "dataset.1")]


def test_select_with_custom_and_selector(tmp_path):
    names = _test_select_with_custom_helper(tmp_path, collection="custom", training_type=TrainingType.UNSUPERVISED)
    assert names == []
    names = _test_select_with_custom_helper(tmp_path, dataset="dataset.1.train", training_type=TrainingType.UNSUPERVISED)
    assert names == []


def test_select_with_custom_only_selector(tmp_path):
    names = _test_select_with_custom_helper(tmp_path, training_type=TrainingType(nab_record.train_type))
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_get_dataset_methods(tmp_path):
    _fill_file(tmp_path, lines=[content_test])
    # write example data
    with open(tmp_path / "path_test.csv", "w") as f:
        f.write(dataset_content)

    dm = DatasetManager(data_folder=tmp_path)
    # get_path
    test_path = dm.get_dataset_path((test_record.collection_name, test_record.dataset_name))
    assert test_path == tmp_path / test_record.test_path
    train_path = dm.get_dataset_path((test_record.collection_name, test_record.dataset_name), train=True)
    assert train_path == tmp_path / test_record.train_path

    # get_df
    test_df = dm.get_dataset_df((test_record.collection_name, test_record.dataset_name))
    pd.testing.assert_frame_equal(test_df, dataset_df, check_datetimelike_compat=True)

    # get ndarray
    test_ndarray = dm.get_dataset_ndarray((test_record.collection_name, test_record.dataset_name))
    assert test_ndarray.shape == dataset_ndarray.shape
    np.testing.assert_equal(test_ndarray, dataset_ndarray)


def test_get_dataset_df_datetime_parsing():
    dm = DatasetManager(data_folder="tests/example_data")
    df = dm.get_dataset_df(("test", "dataset-datetime"))
    assert df["timestamp"].dtype == np.dtype("<M8[ns]")
    df = dm.get_dataset_df(("test", "dataset-int"))
    assert df["timestamp"].dtype == np.dtype("int")


def test_get_dataset_path_missing(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path((nab_record.collection_name, "unknown"))
        assert "not found" in str(ex.value)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path(("custom", "unknown"))
        assert "not found" in str(ex.value)


def test_get_dataset_path_missing_path(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path((nab_record.collection_name, nab_record.dataset_name), train=True)
        assert "not found" in str(ex.value)


def test_initialize_custom_datasets(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file="tests/example_data/datasets.json")
    assert list(dm.get_collection_names()) == ["custom", "NAB"]
    assert list(dm.get_dataset_names()) == custom_dataset_names + ["art_daily_no_noise"]


def test_load_custom_datasets(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]

    dm.load_custom_datasets("tests/example_data/datasets.json")
    assert list(dm.get_collection_names()) == ["custom", "NAB"]
    assert list(dm.get_dataset_names()) == custom_dataset_names + ["art_daily_no_noise"]


def test_df(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    df = dm.df()
    assert len(df) == 1
    assert list(df.index.get_level_values(0)) == [nab_record.collection_name]
    assert df.loc[(nab_record.collection_name, nab_record.dataset_name), "train_type"] == nab_record.train_type


def test_df_with_custom(tmp_path):
    _fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file="tests/example_data/datasets.json")
    df = dm.df()
    assert len(df) == 5
    assert list(df.index.get_level_values(0)) == [nab_record.collection_name] + 4 * ["custom"]
    assert df.loc[(nab_record.collection_name, nab_record.dataset_name), "train_type"] == nab_record.train_type
    assert np.isnan(df.loc[("custom", "dataset.1"), "train_type"])
    assert df.loc[("custom", "dataset.1.train"), "train_path"] == "tests/example_data/dataset.train.csv"


def test_metadata(tmp_path):
    _fill_file(tmp_path, lines=[content_test])
    # write example data
    with open(tmp_path / "path_test.csv", "w") as f:
        f.write(dataset_content)
    with open(tmp_path / "path_train.csv", "w") as f:
        f.write(dataset_content)

    metadata_file = tmp_path / f"{test_record.dataset_name}.metadata.json"
    dm = DatasetManager(tmp_path)
    # test generate file
    metadata = dm.get_detailed_metadata((test_record.collection_name, test_record.dataset_name), train=False)
    assert metadata_file.exists()

    # test generate for train file and add to existing file
    before_size = metadata_file.stat().st_size
    metadata_train = dm.get_detailed_metadata((test_record.collection_name, test_record.dataset_name), train=True)
    assert before_size < metadata_file.stat().st_size
    assert metadata != metadata_train

    # test reading existing file (without changing it)
    before_changed_time = metadata_file.stat().st_mtime_ns
    metadata2 = dm.get_detailed_metadata((test_record.collection_name, test_record.dataset_name), train=False)
    assert before_changed_time == metadata_file.stat().st_mtime_ns
    assert metadata == metadata2
