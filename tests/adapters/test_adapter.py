import unittest
from typing import Optional

import numpy as np

from timeeval.adapters.base import Adapter
from timeeval.data_types import ExecutionType, AlgorithmParameter


class TestBaseAdapter(unittest.TestCase):
    def test_type_error(self):
        with self.assertRaises(TypeError):
            Adapter()

    def test_execution_type_arg(self):
        class TestAdapter(Adapter):
            def __init__(self):
                super().__init__()
                self.execution_type: Optional[ExecutionType] = None

            def _call(self, dataset: AlgorithmParameter, args: dict) -> AlgorithmParameter:
                self.execution_type = args.get("executionType", None)
                return dataset

        a = TestAdapter()
        a(np.empty(1))
        self.assertEqual(a.execution_type, ExecutionType.EXECUTE)
        a(np.empty(1), {"executionType": ExecutionType.TRAIN})
        self.assertEqual(a.execution_type, ExecutionType.TRAIN)
