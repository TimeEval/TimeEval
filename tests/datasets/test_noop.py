import unittest

from timeeval.datasets.custom_noop import NoOpCustomDatasets


class TestNoOpCustomDatasets(unittest.TestCase):

    def test_get_collection_names(self):
        cd = NoOpCustomDatasets()
        assert cd.get_collection_names() == []

    def test_get_dataset_names(self):
        cd = NoOpCustomDatasets()
        assert cd.get_dataset_names() == []

    def test_get_path(self):
        cd = NoOpCustomDatasets()
        with self.assertRaises(KeyError) as cm:
            cd.get_path("dataset.1", train=False)
        assert "No custom datasets loaded!" in str(cm.exception)

    def test_noop(self):
        cd = NoOpCustomDatasets()
        with self.assertRaises(KeyError) as cm:
            cd.get("dataset.1")
        assert "No custom datasets loaded!" in str(cm.exception)
