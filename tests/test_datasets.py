import pytest
import os
import numpy as np

from timeeval.datasets import Datasets, DatasetRecord

header = "collection_name,dataset_name,train_path,test_path,type,datetime_index,split_at,train_type,train_is_normal,input_type,length"
content_nab = "NAB,art_daily_no_noise,,data-processed/univariate/NAB/art_daily_no_noise.test.csv,synthetic,True,,unsupervised,False,univariate,4032"
content_test = "test-collection,test_dataset,path_train.csv,path_test.csv,real,True,,supervised,True,univariate,12"
nab_record = DatasetRecord(
    collection_name="NAB",
    dataset_name="art_daily_no_noise",
    train_path=None,
    test_path="data-processed/univariate/NAB/art_daily_no_noise.test.csv",
    type="synthetic",
    datetime_index=True,
    split_at=np.nan,
    train_type="unsupervised",
    train_is_normal=False,
    input_type="univariate",
    length=4032
)
test_record = DatasetRecord(
    collection_name="test-collection",
    dataset_name="test_dataset",
    train_path="path_train.csv",
    test_path="path_test.csv",
    type="real",
    datetime_index=True,
    split_at=np.nan,
    train_type="supervised",
    train_is_normal=True,
    input_type="univariate",
    length=12
)


def _fill_file(path, lines=(content_nab,)):
    with open(path / Datasets.FILENAME, "w") as f:
        f.write(header)
        f.write("\n")
        for l in lines:
            f.write(l)
            f.write("\n")


def _read_file(path):
    assert os.path.isfile(path / Datasets.FILENAME)
    with open(path / Datasets.FILENAME, "r") as f:
        return [l.strip() for l in f.readlines()]


def _add_dataset(dm):
    dm.add_dataset(
        collection_name=test_record.collection_name,
        dataset_name=test_record.dataset_name,
        train_path=test_record.train_path,
        test_path=test_record.test_path,
        dataset_type=test_record.type,
        datetime_index=test_record.datetime_index,
        split_at=test_record.split_at,
        train_type=test_record.train_type,
        train_is_normal=test_record.train_is_normal,
        input_type=test_record.input_type,
        dataset_length=test_record.length
    )


def test_initialize_empty_folder(tmp_path):
    Datasets(data_folder=tmp_path)
    assert os.path.isfile(tmp_path / Datasets.FILENAME)


def test_initialize_missing_folder(tmp_path):
    target = tmp_path / "subfolder"
    Datasets(data_folder=target)
    assert os.path.isfile(target / Datasets.FILENAME)


def test_initialize_existing(tmp_path):
    _fill_file(tmp_path)
    dm = Datasets(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def test_add_dataset(tmp_path):
    dm = Datasets(data_folder=tmp_path)
    _add_dataset(dm)
    assert list(dm.get_collection_names()) == ["test-collection"]
    assert list(dm.get_dataset_names()) == ["test_dataset"]

    with pytest.raises(Exception) as ex:
        dm.refresh()
        assert "unsaved changes" in str(ex.value)
    lines = _read_file(tmp_path)
    assert len(lines) == 1
    assert lines[0] == header


def test_add_batch_datasets(tmp_path):
    dm = Datasets(data_folder=tmp_path)
    dm.add_datasets([test_record, nab_record])
    dm.save()
    assert list(dm.get_collection_names()) == ["test-collection", "NAB"]
    assert list(dm.get_dataset_names()) == ["test_dataset", "art_daily_no_noise"]

    lines = _read_file(tmp_path)
    assert len(lines) == 3
    assert lines[0] == header
    assert lines[1] == content_test
    assert lines[2] == content_nab


def test_add_dataset_and_save(tmp_path):
    dm = Datasets(data_folder=tmp_path)
    _add_dataset(dm)
    dm.save()
    lines = _read_file(tmp_path)
    assert len(lines) == 2
    assert lines[0] == header
    assert lines[1] == content_test


def test_add_dataset_and_save_with_existing_content(tmp_path):
    _fill_file(tmp_path)
    dm = Datasets(data_folder=tmp_path)
    _add_dataset(dm)
    dm.save()
    lines = _read_file(tmp_path)
    assert len(lines) == 3
    assert lines[0] == header
    assert lines[1] == content_nab
    assert lines[2] == content_test


def test_refresh(tmp_path):
    dm = Datasets(data_folder=tmp_path)
    assert list(dm.get_collection_names()) == []
    assert list(dm.get_dataset_names()) == []

    with open(tmp_path / Datasets.FILENAME, "a") as f:
        f.write(content_nab)
        f.write("\n")

    dm.refresh()
    assert list(dm.get_collection_names()) == ["NAB"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise"]


def _test_select_helper(path, **kwargs):
    _fill_file(path, lines=[content_nab, content_test])
    dm = Datasets(data_folder=path)
    return dm.select(**kwargs)


def test_select_collection(tmp_path):
    names = _test_select_helper(tmp_path, collection_name=nab_record.collection_name)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_name(tmp_path):
    names = _test_select_helper(tmp_path, dataset_name=nab_record.dataset_name)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_dataset_type(tmp_path):
    names = _test_select_helper(tmp_path, dataset_type=nab_record.type)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_datetime_index(tmp_path):
    names = _test_select_helper(tmp_path, datetime_index=nab_record.datetime_index)
    assert names == [(nab_record.collection_name, nab_record.dataset_name),
                     (test_record.collection_name, test_record.dataset_name)]


def test_select_train_type(tmp_path):
    names = _test_select_helper(tmp_path, train_type=nab_record.train_type)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_train_is_normal(tmp_path):
    names = _test_select_helper(tmp_path, train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]


def test_select_input_type(tmp_path):
    names = _test_select_helper(tmp_path, input_type=nab_record.input_type)
    assert names == [(nab_record.collection_name, nab_record.dataset_name),
                     (test_record.collection_name, test_record.dataset_name)]


def test_select_combined(tmp_path):
    names = _test_select_helper(tmp_path,
                                collection_name=nab_record.collection_name,
                                train_is_normal=nab_record.train_is_normal)
    assert names == [(nab_record.collection_name, nab_record.dataset_name)]
