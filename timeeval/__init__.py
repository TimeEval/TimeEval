"""
TimeEval
========

An Evaluation Tool for Anomaly Detection Algorithms on Time Series Data.

Provides:
  1. tbd

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided with the code, and a loose standing reference guide,
available on `timeeval.readthedocs.io <timeeval.readthedocs.io>`_.

Code snippets in the docstring examples are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's or class' docstring::

  >>> from timeeval import TimeEval
  >>> help(TimeEval)
  ... # doctest: +SKIP

Available subpackages
---------------------
adapters
    Algorithm adapters that allow TimeEval to execute various algorithms.
datasets
    Dataset manager API to build and load dataset collections and extract metadata about them.
heuristics
    Methods to set algorithm hyperparameters heuristically, for example based on dataset metadata.
params
    Hyperparameter search definitions for algorithms.
utils
    Some utility functions, such as quality metrics or window operations.

Utilities
---------
__version__
    TimeEval version string
TrainingType
    Definition of an algorithms' training type.
InputDimensionality
    Definition of an algorithms' input dimensionality.
"""

from ._version import __version__
from .algorithm import Algorithm
from .data_types import AlgorithmParameter, TrainingType, InputDimensionality
from .datasets import Datasets, DatasetManager, MultiDatasetManager
from .remote_configuration import RemoteConfiguration
from .resource_constraints import ResourceConstraints
from .timeeval import TimeEval, Status
from .utils.metrics import Metric, DefaultMetrics
