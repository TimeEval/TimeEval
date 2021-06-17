import contextlib
from typing import Generator

import joblib
from joblib.parallel import BatchCompletionCallBack
from tqdm import tqdm


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator[tqdm, None, None]:
    """Context manager to patch joblib to report into tqdm progress bar given as argument.

    Directly taken from https://stackoverflow.com/a/58936697.

    Examples
    --------
    >>> import time
    >>> from joblib import Parallel, delayed
    >>>
    >>> def some_method(wait_time):
    >>>     time.sleep(wait_time)
    >>>
    >>> with tqdm_joblib(tqdm(desc="Sleeping method", total=10)):
    >>>     Parallel(n_jobs=2)(delayed(some_method)(0.2) for i in range(10))
    """

    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
