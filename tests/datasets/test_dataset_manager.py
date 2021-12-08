import pytest

from tests.fixtures.dataset_fixtures import add_dataset, read_file, fill_file, dataset_index_header, test_record, \
    nab_record, dataset_index_content_nab, dataset_index_content_test
from timeeval import DatasetManager


def test_add_dataset(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    add_dataset(dm)
    assert list(dm.get_collection_names()) == ["test-collection"]
    assert list(dm.get_dataset_names()) == ["test_dataset"]

    with pytest.raises(Exception) as ex:
        dm.refresh()
        assert "unsaved changes" in str(ex.value)
    lines = read_file(tmp_path)
    assert len(lines) == 1
    assert lines[0] == dataset_index_header


def test_add_batch_datasets(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    dm.add_datasets([test_record, nab_record])
    dm.save()
    # entries are internally sorted to allow fast MultiIndex-Slicing!
    assert list(dm.get_collection_names()) == ["NAB", "test-collection"]
    assert list(dm.get_dataset_names()) == ["art_daily_no_noise", "test_dataset"]

    lines = read_file(tmp_path)
    assert len(lines) == 3
    assert lines[0] == dataset_index_header
    assert lines[1] == dataset_index_content_nab
    assert lines[2] == dataset_index_content_test


def test_add_dataset_and_save(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    add_dataset(dm)
    dm.save()
    lines = read_file(tmp_path)
    assert len(lines) == 2
    assert lines[0] == dataset_index_header
    assert lines[1] == dataset_index_content_test


def test_add_dataset_and_save_with_existing_content(tmp_path):
    fill_file(tmp_path)
    dm = DatasetManager(data_folder=tmp_path)
    add_dataset(dm)
    dm.save()
    lines = read_file(tmp_path)
    assert len(lines) == 3
    assert lines[0] == dataset_index_header
    assert lines[1] == dataset_index_content_nab
    assert lines[2] == dataset_index_content_test


def test_context_manager(tmp_path):
    fill_file(tmp_path)
    with DatasetManager(data_folder=tmp_path) as dm:
        add_dataset(dm)
    lines = read_file(tmp_path)
    assert len(lines) == 3
    assert lines[0] == dataset_index_header
    assert lines[1] == dataset_index_content_nab
    assert lines[2] == dataset_index_content_test
