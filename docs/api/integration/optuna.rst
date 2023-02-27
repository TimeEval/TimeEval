Optuna integration
==================

`Optuna <https://optuna.org>`_ is an automatic hyperparameter optimization framework and this integration allows you to
use it within TimeEval. TimeEval will load the :class:`~timeeval.integration.optuna.OptunaModule` automatically if
at least one algorithm uses :class:`~timeeval.params.BayesianParameterSearch` as its parameter search strategy. Please
make sure that you install the required dependencies for Optuna before using this integration (we also recommend to
install `psycopg2` to use the PostgreSQL storage backend):

.. code-block:: bash

   pip install 'optuna>=3.1.0' psycopg2

The following Optuna features are supported:

- Definition of search spaces using Optuna distributions for each algorithm (one study per algorithm):
  :class:`~timeeval.params.BayesianParameterSearch`.
- Configurable `samplers <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html>`_.
- Configurable `storage backends <https://optuna.readthedocs.io/en/stable/reference/storages.html>`_ (in-memory, RDB,
  Journal, etc.).
- `Resuming <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html>`_ of existing studies (via RDB
  storage backend).
- Parallel and distributed parameter search of a single or multiple studies (synchronized via RDB storage backend).


.. warning::
    Parameter search using the Optuna integration is **non-deterministic**. The results may vary between different runs,
    even if the same seed is used (e.g., for the Optuna sampler or pruner). This is because TimeEval needs to re-seed
    the Optuna samplers for every trial in distributed mode. This is necessary to ensure that initial random samples are
    different over all workers.

TimeEval will automatically manage an RDB storage backend if you use the default configuration. This allows you to start
TimeEval in distributed mode and perform the parameter search in parallel and distributed.

timeeval.integration.optuna.OptunaModule
----------------------------------------

.. autoclass:: timeeval.integration.optuna.OptunaModule
   :members:
   :undoc-members:
   :show-inheritance:

timeeval.integration.optuna.OptunaConfiguration
-----------------------------------------------

.. autoclass:: timeeval.integration.optuna.OptunaConfiguration
   :members:
   :undoc-members:
   :show-inheritance:

timeeval.integration.optuna.OptunaStudyConfiguration
----------------------------------------------------

.. autoclass:: timeeval.integration.optuna.OptunaStudyConfiguration
   :members:
   :undoc-members:
   :show-inheritance:
