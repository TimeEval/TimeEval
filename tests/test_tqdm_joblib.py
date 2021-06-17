import unittest

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from timeeval.utils.tqdm_joblib import tqdm_joblib


class TestTqdmJoblib(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @staticmethod
    def _method(x: int) -> int:
        return x + 1

    def test_wraps_joblib_correctly(self):
        total = 5
        with tqdm_joblib(tqdm(desc="Joblib tqdm wrapper", total=total)):
            self.assertEqual(joblib.parallel.BatchCompletionCallBack.__name__, "TqdmBatchCompletionCallback")
            res = Parallel()(
                delayed(self._method)(i) for i in range(5)
            )
        self.assertEqual(joblib.parallel.BatchCompletionCallBack.__name__, "BatchCompletionCallBack")
        self.assertListEqual(res, list(map(self._method, range(5))))
