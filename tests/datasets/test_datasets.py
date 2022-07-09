from pathlib import Path

import pytest
import os

import numpy as np
import pandas as pd

from tests.fixtures.dataset_fixtures import fill_file, dataset_index_content_nab, dataset_index_content_test, \
    nab_record, test_record, custom_dataset_names, dataset_content, dataset_df, dataset_ndarray, CUSTOM_DATASET_PATH, \
    dataset_metadata
from timeeval import Datasets, DatasetManager, InputDimensionality, TrainingType


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
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def test_refresh(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == []
    assert list(dm.get_dataset_names()) == []

    with open(tmp_path / Datasets.INDEX_FILENAME, "a") as f:
        f.write(dataset_index_content_nab)
        f.write("\n")

    dm.refresh()
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def _test_select_helper(path, lines=(dataset_index_content_nab, dataset_index_content_test), **kwargs):
    fill_file(path, lines)
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


def test_select_min_anomalies(tmp_path):
    names = _test_select_helper(tmp_path, min_anomalies=2)
    assert names == [(test_record.collection_name, test_record.dataset_name)]


def test_select_max_anomalies(tmp_path):
    names = _test_select_helper(tmp_path, max_anomalies=1)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_max_contamination(tmp_path):
    names = _test_select_helper(tmp_path, max_contamination=0.01)
    assert names == [(test_record.collection_name, test_record.dataset_name)]


def test_select_combined(tmp_path):
    names = _test_select_helper(tmp_path,
                                collection=nab_record.collection_name,
                                train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_wrong_order(tmp_path):
    names = _test_select_helper(tmp_path,
                                lines=[dataset_index_content_test, dataset_index_content_nab],
                                collection=test_record.collection_name,
                                train_is_normal=test_record.train_is_normal)
    assert names == [(test_record.collection_name, test_record.dataset_name)]


def _test_select_with_custom_helper(path, **kwargs):
    fill_file(path)
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
    assert names == [("custom", "dataset.1"), ("custom", "dataset.3"), ("custom", "dataset.4")]
    names = _test_select_with_custom_helper(tmp_path, dataset="dataset.1.train", training_type=TrainingType.UNSUPERVISED)
    assert names == []


def test_select_with_custom_only_selector(tmp_path):
    names = _test_select_with_custom_helper(tmp_path, max_contamination=0.01)
    assert names == [('custom', 'dataset.1'), ('custom', 'dataset.1.train'),
                     ('custom', 'dataset.3'), ('custom', 'dataset.4')]
    names = _test_select_with_custom_helper(tmp_path, input_dimensionality=InputDimensionality.MULTIVARIATE)
    assert names == [('custom', 'dataset.4')]


def test_select_with_custom_ignores_datetime_index(tmp_path):
    names = _test_select_with_custom_helper(tmp_path, datetime_index=True)
    # the custom datasets do not contain information about the index type and therefore ignore the
    # 'datetime_index'-selector (see warning in output)
    assert names == [(nab_record.collection_name, nab_record.dataset_name), ('custom', 'dataset.1'),
                     ('custom', 'dataset.1.train'), ('custom', 'dataset.3'), ('custom', 'dataset.4')]


def test_get(tmp_path):
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file=CUSTOM_DATASET_PATH)
    dataset = dm.get(("custom", "dataset.1"))
    assert dataset.datasetId == ("custom", "dataset.1")
    assert dataset.training_type == TrainingType.UNSUPERVISED
    assert dataset.length == dataset_metadata.length
    assert dataset.dimensions == dataset_metadata.dimensions
    assert dataset.num_anomalies == dataset_metadata.num_anomalies
    assert dataset.max_anomaly_length == dataset_metadata.anomaly_length.max
    assert dataset.median_anomaly_length == dataset_metadata.anomaly_length.median
    assert dataset.min_anomaly_length == dataset_metadata.anomaly_length.min

    # other ways to call get
    dataset1 = dm.get("custom", "dataset.1")
    assert dataset == dataset1
    with pytest.raises(ValueError) as ex:
        dm.get("dataset.1")
    assert "Cannot use" in str(ex.value)


def test_get_dataset_methods(tmp_path):
    fill_file(tmp_path, lines=[dataset_index_content_test])
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


def test_get_dataset_methods_custom(tmp_path):
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file=CUSTOM_DATASET_PATH)
    dataset_id = ("custom", "dataset.1.train")
    # get_path
    test_path = dm.get_dataset_path(dataset_id)
    assert test_path == Path("./tests/example_data/dataset.test.csv").resolve()
    train_path = dm.get_dataset_path(dataset_id, train=True)
    assert train_path == tmp_path / Path("./tests/example_data/dataset.train.csv").resolve()

    # get_df
    df = dm.get_dataset_df(dataset_id)
    dataset_df = pd.read_csv(Path("./tests/example_data/dataset.test.csv").resolve(),
                             parse_dates=["timestamp"],
                             infer_datetime_format=True)
    pd.testing.assert_frame_equal(df, dataset_df, check_datetimelike_compat=True)

    # get_ndarray
    test_ndarray = dm.get_dataset_ndarray(dataset_id)
    assert test_ndarray.shape == dataset_df.shape
    np.testing.assert_equal(test_ndarray, dataset_df.values)


def test_get_dataset_df_datetime_parsing():
    dm = DatasetManager(data_folder="tests/example_data", custom_datasets_file=CUSTOM_DATASET_PATH)
    # bundled datasets
    df = dm.get_dataset_df(("test", "dataset-datetime"))
    assert df["timestamp"].dtype == np.dtype("<M8[ns]")
    df = dm.get_dataset_df(("test", "dataset-int"))
    assert df["timestamp"].dtype == np.dtype("int64")
    # custom datasets
    df = dm.get_dataset_df(("custom", "dataset.1.train"), train=False)
    assert df["timestamp"].dtype == np.dtype("<M8[ns]")
    df = dm.get_dataset_df(("custom", "dataset.1.train"), train=True)
    assert df["timestamp"].dtype == np.dtype("int64")


def test_get_dataset_path_missing(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path((nab_record.collection_name, "unknown"))
    assert "not found" in str(ex.value)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path(("custom", "unknown"))
    assert "not found" in str(ex.value)


def test_get_dataset_path_missing_path(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    with pytest.raises(KeyError) as ex:
        dm.get_dataset_path((nab_record.collection_name, nab_record.dataset_name), train=True)
    assert "not found" in str(ex.value)


def test_initialize_custom_datasets(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file="tests/example_data/datasets.json")
    assert list(dm.get_collection_names()) == ["custom", "NAB"]
    assert list(dm.get_dataset_names()) == custom_dataset_names + ["art_daily_no_noise"]


def test_load_custom_datasets(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]

    dm.load_custom_datasets("tests/example_data/datasets.json")
    assert list(dm.get_collection_names()) == ["custom", "NAB"]
    assert list(dm.get_dataset_names()) == custom_dataset_names + ["art_daily_no_noise"]


def test_df(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    df = dm.df()
    assert len(df) == 1
    assert list(df.index.get_level_values(0)) == [nab_record.collection_name]
    assert df.loc[(nab_record.collection_name, nab_record.dataset_name), "train_type"] == nab_record.train_type


def test_df_with_custom(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path, custom_datasets_file="tests/example_data/datasets.json")
    df = dm.df()
    assert len(df) == 5
    assert list(df.index.get_level_values(0)) == [nab_record.collection_name] + 4 * ["custom"]
    assert list(df.index.get_level_values(1)) == [
        nab_record.dataset_name, "dataset.1", "dataset.1.train", "dataset.3", "dataset.4"
    ]
    assert df.loc[(nab_record.collection_name, nab_record.dataset_name), "train_type"] == nab_record.train_type
    assert np.isnan(df.loc[("custom", "dataset.1"), "stddev"])
    assert np.isnan(df.loc[("custom", "dataset.1.train"), "mean"])
    assert np.isnan(df.loc[("custom", "dataset.3"), "period_size"])
    assert df.loc[("custom", "dataset.1.train"), "train_path"] == str(Path("tests/example_data/dataset.train.csv").resolve())


def test_metadata(tmp_path):
    fill_file(tmp_path, lines=[dataset_index_content_test])
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


def test_metadata_custom(tmp_path):
    dm = DatasetManager(tmp_path, custom_datasets_file=CUSTOM_DATASET_PATH)
    meta = dm.get(("custom", "dataset.1"))
    metadata = dm.get_detailed_metadata(("custom", "dataset.1"), train=False)

    # nothing was written to disk
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files == [tmp_path / "datasets.csv"]

    # metadata is correct
    assert metadata.channels == meta.dimensions
    assert metadata.length == meta.length
    assert metadata.shape == (meta.length, meta.dimensions)
    assert metadata.num_anomalies == meta.num_anomalies
    np.testing.assert_array_almost_equal(list(metadata.means.values()), [15836.7119], decimal=4)


def test_get_training_type():
    dm = DatasetManager(data_folder="tests/example_data", custom_datasets_file=CUSTOM_DATASET_PATH)
    tt = dm.get_training_type(("test", "dataset-datetime"))
    assert tt == TrainingType.SUPERVISED
    tt = dm.get_training_type(("custom", "dataset.1"))
    assert tt == TrainingType.UNSUPERVISED


def test_get_training_type_ambiguous(tmp_path):
    fill_file(tmp_path, lines=[
        "test,dataset-semi-supervised,dataset.train.csv,dataset.test.csv,synthetic,True,,semi-supervised,False,univariate,0,1,0.01,2,5,5,6,564.52,2.468,no trend,not_stationary,12"
    ])
    dm = DatasetManager(tmp_path)
    tt = dm.get_training_type(("test", "dataset-semi-supervised"))
    # should fall back to SUPERVISED if train_is_normal is False
    assert tt == TrainingType.SUPERVISED
