.. TimeEval documentation master file, created by
   sphinx-quickstart on Mon Jul 11 09:00:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TimeEval: Evaluation Tool for Anomaly Detection Algorithms on Time Series
=========================================================================

.. toctree::
   :maxdepth: 1
   :hidden:

   user/index
   concepts/index
   api/index
   dev/index


Overview
--------

TimeEval is an evaluation tool for time series anomaly detection algorithms.
It defines common interfaces for datasets and algorithms to allow the efficient comparison of the algorithms' quality and runtime performance.
TimeEval can be configured using a simple Python API and comes with
`a large collection of compatible datasets <https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html>`_
and `algorithms <https://github.com/HPI-Information-Systems/TimeEval-algorithms>`_.

TimeEval takes your input and automatically creates experiment configurations by taking the cross-product of your inputs.
It executes all experiment configurations one after the other or - when distributed - in parallel and records the anomaly detection quality and the runtime of the algorithms.

TimeEval takes four inputs for the experiment creation:

1. Algorithms
2. Datasets
3. Algorithm ParameterGrids
4. A repetition number

The following code snippet shows a simple example experiment evaluating `LOF <https://github.com/HPI-Information-Systems/TimeEval-algorithms/tree/main/lof>`_
and a simple baseline algorithm on some test data:

.. literalinclude:: ../easy-example-experiment.py
   :language: python
   :name: example-experiment-py
   :caption: Example usage of TimeEval
   :linenos:

Features
^^^^^^^^

Listing :ref:`example-experiment-py` illustrates some of the main features of TimeEval:

Dataset API:
   :doc:`Interface <api/timeeval.datasets>` to available dataset collection to select datasets easily (L19-20).
Algorithm Adapter Architecture:
   TimeEval supports different algorithm adapters to execute simple Python functions or whole pipelines and applications (L25, L35).
Hyperparameter Specification:
   Algorithm hyperparameters can be specified using different search grids (L26-29).
Metrics:
   TimeEval provides various evaluation metrics (such as :attr:`timeeval.utils.metrics.DefaultMetrics.ROC_AUC`,
   :attr:`timeeval.utils.metrics.DefaultMetrics.RANGE_PR_AUC`, or :attr:`timeeval.utils.metrics.FScoreAtK`)
   and measures algorithm runtimes automatically (L42).
Distributed Execution:
   TimeEal can be deployed on a compute cluster to execute evaluation tasks distributedly.

Installation
^^^^^^^^^^^^

Prerequisites:

- Python 3.7, 3.8, or 3.9
- `Docker <https://www.docker.com>`_
- `rsync <https://rsync.samba.org>`_ (if you want to use distributed TimeEval)

TimeEval is published to `PyPI <https://pypi.org>`_ and you can install it using:

.. code-block:: bash

   pip install TimeEval

.. attention::

   Currently TimeEval is tested only on Linux systems and relies on unixoid capabilities.


License
^^^^^^^

The project is licensed under the `MIT license <https://mit-license.org>`_.

If you use TimeEval in your project or research, please cite our demonstration paper:

   Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock.
   TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB, 15(12): 3678 - 3681, 2022.
   doi:`10.14778/3554821.3554873 <https://doi.org/10.14778/3554821.3554873>`_

You can use the following BibTex entry:

.. code-block:: bibtex

   @article{WenigEtAl2022TimeEval,
     title = {TimeEval: {{A}} Benchmarking Toolkit for Time Series Anomaly Detection Algorithms},
     author = {Wenig, Phillip and Schmidl, Sebastian and Papenbrock, Thorsten},
     date = {2022},
     journaltitle = {Proceedings of the {{VLDB Endowment}} ({{PVLDB}})},
     volume = {15},
     number = {12},
     pages = {3678--3681},
     doi = {10.14778/3554821.3554873}
   }


User Guide
----------

New to TimeEval? Check out our :doc:`/user/index` to get started with TimeEval.
The user guides explain TimeEval's API and how to use it to achieve your goal.

:doc:`To the user guide</user/index>`


TimeEval Concepts
-----------------

Background information and in-depth explanations about how TimeEval works can be found in the
:doc:`TimeEval concepts reference</concepts/index>`.

:doc:`To the concepts</concepts/index>`


API Reference
-------------

The API reference guide contains a detailed description of the functions, modules, and objects included in TimeEval. The
API reference describes how the methods work and which parameters can be used.

:doc:`To the API reference</api/index>`


Contributor's Guide
-------------------

Want to add to the codebase? You can help with the documentation? The contributing guidelines will guide you through the
process of improving TimeEval and its ecosystem.

:doc:`To the contributor's guide</dev/index>`


Additional Links
================

* `TimeEval Github repository <https://github.com/HPI-Information-Systems/TimeEval>`_
* `TimeEval algorithms <https://github.com/HPI-Information-Systems/TimeEval-algorithms>`_
* `Datasets <https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html>`_ for TimeEval
* `TimeEval GUI <https://github.com/HPI-Information-Systems/TimeEval-GUI>`_ (prototype)
* Time series anomaly dataset generator `GutenTAG <https://github.com/HPI-Information-Systems/gutentag>`_
* :ref:`modindex`
* :ref:`search`
