import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Callable, Tuple, Any
import numpy as np

from .algorithm import Algorithm
from .data_types import AlgorithmParameter


@dataclass
class Times:
    main: float
    preprocess: Optional[float] = None
    postprocess: Optional[float] = None

    def to_dict(self) -> Dict:
        return {f"{k}_time": v for k, v in asdict(self).items()}

    @staticmethod
    def from_algorithm(algorithm: Algorithm, X: AlgorithmParameter, args: dict) -> Tuple[np.ndarray, 'Times']:
        x, pre_time = timer(algorithm.preprocess, X, args) if algorithm.preprocess else (X, None)
        x, main_time = timer(algorithm.main, x, args)
        x, post_time = timer(algorithm.postprocess, x, args) if algorithm.postprocess else(x, None)
        return x, Times(main_time, preprocess=pre_time, postprocess=post_time)


def timer(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    fn_result = fn(*args, **kwargs)
    end = time.time()
    duration = end - start
    return fn_result, duration
