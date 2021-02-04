import unittest

from timeeval.adapters.base import Adapter


class TestBaseAdapter(unittest.TestCase):
    def test_type_error(self):
        with self.assertRaises(TypeError):
            Adapter()
