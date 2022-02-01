from pathlib import Path

import pytest

from tests.fixtures.dataset_fixtures import add_dataset
from timeeval import DatasetManager, MultiDatasetManager


def _prepare(tmp_path):
    dm = DatasetManager(data_folder=tmp_path)
    add_dataset(dm)
    dm.save()


def test_load_multiple_folders(tmp_path):
    _prepare(tmp_path)
    dmgr = MultiDatasetManager(["./tests/example_data", tmp_path])
    assert list(dmgr.get_collection_names()) == ["test", "test-collection"]
    assert list(dmgr.get_dataset_names()) == ["dataset-datetime", "dataset-int", "test_dataset"]


def test_dataset_path(tmp_path):
    base_path_example = Path("./tests/example_data")
    base_path_tmp = Path(tmp_path)
    _prepare(tmp_path)
    dmgr = MultiDatasetManager([base_path_example, base_path_tmp])

    assert dmgr.get_dataset_path(("test", "dataset-int"), train=False) == (
                base_path_example.resolve() / "dataset.train.csv"
    )
    assert dmgr.get_dataset_path(("test-collection", "test_dataset"), train=False) == (
                base_path_tmp.resolve() / "path_test.csv"
    )


def test_missing_dataset_path(tmp_path):
    target = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as ex:
        MultiDatasetManager(["./tests/example_data", target])
    assert "Could not find the index files" in str(ex.value)
    assert "missing" in str(ex.value)
