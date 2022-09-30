import datetime as dt
import logging
import signal
import socket
import subprocess
from enum import Enum
from pathlib import Path
from time import time
from types import FrameType
from typing import Callable, List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import tqdm
from distributed.client import Future
from joblib import Parallel, delayed

from .adapters.docker import DockerTimeoutError, DockerAdapter
from .algorithm import Algorithm
from .constants import RESULTS_CSV
from .core.experiments import Experiments, Experiment
from .core.remote import Remote, RemoteConfiguration
from .core.times import Times
from .data_types import TrainingType, InputDimensionality
from .datasets import Datasets
from .resource_constraints import ResourceConstraints
from .utils.encode_params import dumps_params
from .metrics import Metric, DefaultMetrics
from .utils.tqdm_joblib import tqdm_joblib


class Status(Enum):
    """Status of an experiment.

    The status of each evaluation experiment can have one of three states: ok, error, or timeout.
    """
    OK = 0
    ERROR = 1
    TIMEOUT = 2


class TimeEval:
    """Main class of TimeEval.

    This class is the main utility to configure and execute evaluation experiments.
    First select your algorithms and datasets and, then, pass them to TimeEval and use its constructor arguments to
    configure your evaluation run.
    Per default, TimeEval evaluates all algorithms on all datasets (cross product).
    You can use the parameters ``skip_invalid_combinations``, ``force_training_type_match``,
    ``force_dimensionality_match``, and ``experiment_combinations_file`` to control which algorithm runs on which dataset.
    See the description of the other arguments for further configuration details.

    After you have created your TimeEval object, holding the experiment run configuration, you can execute the
    experiments by calling :func:`~timeeval.TimeEval.run`.
    Afterwards, the evaluation summary results are accessible in the ``results_path`` and from
    :func:`~timeeval.TimeEval.get_results`.

    Examples
    --------
    Simple example experiment evaluating a single algorithm on the test datasets using the default metrics (just
    :attr:`~timeeval.utils.metrics.DefaultMetrics.ROC_AUC`):

    >>> from timeeval import TimeEval, DefaultMetrics, Algorithm, TrainingType, InputDimensionality, DatasetManager
    >>> from timeeval.adapters import DockerAdapter
    >>> from timeeval.params import FixedParameters
    >>>
    >>> dm = DatasetManager(Path("tests/example_data"))
    >>> datasets = dm.select()
    >>>
    >>> algorithms = [
    >>>     Algorithm(
    >>>         name="COF",
    >>>         main=DockerAdapter(image_name="registry.gitlab.hpi.de/akita/i/cof"),
    >>>         param_config=FixedParameters({"n_neighbors": 20, "random_state": 42}),
    >>>         data_as_file=True,
    >>>         training_type=TrainingType.UNSUPERVISED,
    >>>         input_dimensionality=InputDimensionality.MULTIVARIATE
    >>>     ),
    >>> ]
    >>>
    >>> timeeval = TimeEval(dm, datasets, algorithms, metrics=DefaultMetrics.default_list())
    >>> timeeval.run()
    >>> results = timeeval.get_results(aggregated=False)
    >>> print(results)

    Parameters
    ----------
    dataset_mgr : ~timeeval.datasets.datasets.Datasets
        The dataset manager provides the metadata about the datasets. You can either use a
        :class:`~timeeval.datasets.dataset_manager.DatasetManager` or a
        :class:`~timeeval.datasets.multi_dataset_manager.MultiDatasetManager`.
    datasets : List[Tuple[str, str]]
        List of dataset IDs consisting of collection name and dataset name to uniquely identify each dataset. The
        datasets must be known by the ``dataset_mgr``. You can call :func:`~timeeval.datasets.datasets.Datasets.select`
        on the ``dataset_mgr`` to get a list of dataset IDs.
    algorithms : List[Algorithm]
        List of algorithms to evaluate on the datasets.
        The algorithm specification also contains the hyperparameter configurations that TimeEval will test.
    results_path : Path
        Use this parameter to change the path where all evaluation results are stored.
        If TimeEval is used in distributed mode, this path will be created on **all** nodes!
    repetitions : int
        Execute each unique combination of dataset, algorithm, and hyperparameter-setting multiple times.
        This allows you to use TimeEval to measure runtimes more precisely by aggregating the runtime measurements over
        multiple repetitions.
    distributed : bool
        Run TimeEval in distributed mode.
        In this case, you **should** also supply a ``remote_config``.
    remote_config : Optional[RemoteConfiguration]
        Configuration of the Dask cluster used for distributed execution of TimeEval.
        See :class:`~timeeval.RemoteConfiguration` for details.
    resource_constraints : Optional[ResourceConstraints]
        You can supply a :class:`~timeeval.ResourceConstraints`-object to limit the amount of (CPU, memory, or runtime)
        resources available to each experiment.
        These options apply to each experiment to ensure a fair comparison.

        .. warning::
            Resource constraints are currently only implemented by the :class:`~timeeval.adapters.docker.DockerAdapter`.
            If you rely on resource constraints, please make sure that all algorithms use the ``DockerAdapter``-implementation.

    disable_progress_bar : bool
        Enable / disable showing the `tqdm <https://tqdm.github.io>`_ progress bars.
    metrics : Optional[List[Metric]]
        Supply a list of :class:`~timeeval.utils.metrics.Metric` to evaluate the algorithms with.
        TimeEval computes all supplied metrics over all experiments.
        If you don't specify any metric (``None``), the default metric list
        :func:`~timeeval.utils.metrics.DefaultMetrics.default_list` is used instead.
    skip_invalid_combinations : bool
        Not all algorithms can be executed on all datasets.
        If this flag is set to ``True``, TimeEval will skip all invalid combinations of algorithms and datasets based on
        their input dimensionality and training type.
        It is automatically enabled if either ``force_training_type_match`` or ``force_dimensionality_match`` is set to
        ``True``.
        Per default (``force_training_type_match == force_dimensionality_match == False``), the following combinations
        are not executed:

        - supervised algorithms on semi-supervised or unsupervised datasets (datasets cannot be used to train the algorithm)
        - semi-supervised algorithm on supervised or unsupervised datasets (datasets cannot be used to train the algorithm)
        - univariate algorithms on multivariate datasets (algorithm cannot process the dataset)

    force_training_type_match : bool
        Narrow down the algorithm-dataset combinations further by executing an algorithm only on datasets with **the same**
        training type, e.g. unsupervised algorithms only on unsupervised datasets.
        This flag implies ``skip_invalid_combinations==True``.
    force_dimensionality_match : bool
        Narrow down the algorithm-dataset combinations furthter by executing an algorithm only on datasets with **the same**
        input dimensionality, e.g. multivariate algorithms only on multivariate datasets.
        This flag implies ``skip_invalid_combinations==True``.
    n_jobs : int
        Set the number of jobs / processes used to fetch the results from the remote machine.
        This setting is used only in distributed mode.
        ``-1`` instructs TimeEval to use all locally available cores.
    experiment_combinations_file : Optional[Path]
        Supply a path to an experiment combinations CSV-File.
        Using this file, you can specify explicitly which combinations of algorithms, datasts, and hyperparameters
        should be executed.
        The file should contain CSV data with a single header line and four columns with the following names:

        1. `algorithm` - name of the algorithm
        2. `collection` - name of the dataset collection
        3. `dataset` - name of the dataset
        4. `hyper_params_id` - ID of the hyperparameter configuration

        Only experiments that are present in the TimeEval configuration **and** this file are scheduled and executed.
        This allows you to circumvent the cross-product that TimeEval will perform in its default configuration.
    """

    RESULT_KEYS = ["algorithm",
                   "collection",
                   "dataset",
                   "algo_training_type",
                   "algo_input_dimensionality",
                   "dataset_training_type",
                   "dataset_input_dimensionality",
                   "train_preprocess_time",
                   "train_main_time",
                   "execute_preprocess_time",
                   "execute_main_time",
                   "execute_postprocess_time",
                   "status",
                   "error_message",
                   "repetition",
                   "hyper_params",
                   "hyper_params_id"]
    """This list contains all the _fixed_ result data frame's column headers.
    TimeEval dynamically adds the metrics and execution times depending on its configuration.
    
    For metrics, their :func:`~timeeval.utils.metrics.Metric.name` will be used as column header, and TimeEval will add
    the following runtime measurements depending on whether they are applicable to the algorithms in the run or not:
    
    - train_preprocess_time: if :func:`~timeeval.Algorithm.preprocess` is defined
    - train_main_time: if the algorithm is semi-supervised or supervised
    - execute_preprocess_time: if :func:`~timeeval.Algorithm.preprocess` is defined
    - execute_main_time: always
    - execute_postprocess_time: if :func:`~timeeval.Algorithm.postprocess` is defined
    """

    DEFAULT_RESULT_PATH = Path("./results")
    """Default path for the results.
    
    If you don't specify the ``results_path``, TimeEval will store the evaluation results in the folder ``results``
    within the current working directory.
    """

    def __init__(self,
                 dataset_mgr: Datasets,
                 datasets: List[Tuple[str, str]],
                 algorithms: List[Algorithm],
                 results_path: Path = DEFAULT_RESULT_PATH,
                 repetitions: int = 1,
                 distributed: bool = False,
                 remote_config: Optional[RemoteConfiguration] = None,
                 resource_constraints: Optional[ResourceConstraints] = None,
                 disable_progress_bar: bool = False,
                 metrics: Optional[List[Metric]] = None,
                 skip_invalid_combinations: bool = True,
                 force_training_type_match: bool = False,
                 force_dimensionality_match: bool = False,
                 n_jobs: int = -1,
                 experiment_combinations_file: Optional[Path] = None) -> None:
        assert len(datasets) > 0, "No datasets given for evaluation!"
        assert len(algorithms) > 0, "No algorithms given for evaluation!"
        assert repetitions > 0, "Negative or 0 repetitions are not supported!"
        assert n_jobs >= -1, f"n_jobs={n_jobs} not supported (must be >= -1)!"
        if experiment_combinations_file is not None:
            assert experiment_combinations_file.exists(), "Experiment combination file not found!"

        dataset_details = []
        not_found_datasets = []
        for d in datasets:
            try:
                dataset_details.append(dataset_mgr.get(d))
            except KeyError:
                not_found_datasets.append(repr(d))
        assert len(not_found_datasets) == 0, "Some datasets could not be found in DatasetManager!\n  " \
                                             f"{', '.join(not_found_datasets)}"

        limits = resource_constraints or ResourceConstraints.default_constraints()
        if limits != ResourceConstraints.default_constraints():
            incompatible_algos = [a.name for a in algorithms if not isinstance(a.main, DockerAdapter)]
            assert len(incompatible_algos) == 0, "The following algorithms won't satisfy the specified resource " \
                                                 f"constraints: {', '.join(incompatible_algos)}. Either drop the " \
                                                 "resource constraints or use the DockerAdapter for all algorithms!"

        self.log = logging.getLogger(self.__class__.__name__)
        start_date: str = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_path = results_path.resolve() / start_date
        self.disable_progress_bar = disable_progress_bar
        self.metrics: List[Metric] = metrics or DefaultMetrics.default_list()
        self.metric_names = [m.name for m in self.metrics]
        self.results = pd.DataFrame(columns=TimeEval.RESULT_KEYS + self.metric_names)
        self.distributed = distributed
        self.n_jobs = n_jobs

        self.log.info(f"Results are recorded in the directory {self.results_path}")
        self.results_path.mkdir(parents=True, exist_ok=True)

        if not distributed and limits.tasks_per_host > 1:
            self.log.warning(
                f"`tasks_per_host` was set to {limits.tasks_per_host}. However, non-distributed "
                "execution of TimeEval does currently not support parallelism! Reducing `tasks_per_host` to 1. "
                "The automatic resource limitation will reflect this by increasing the limits for the single task. "
                "Explicitly set constraints to limit the resources for local executions of TimeEval."
            )
            limits.tasks_per_host = 1

        self.exps = Experiments(dataset_mgr, dataset_details, algorithms, self.results_path,
                                resource_constraints=limits,
                                repetitions=repetitions,
                                skip_invalid_combinations=skip_invalid_combinations,
                                force_training_type_match=force_training_type_match,
                                force_dimensionality_match=force_dimensionality_match,
                                metrics=self.metrics,
                                experiment_combinations_file=experiment_combinations_file)

        self.remote_config: RemoteConfiguration = remote_config or RemoteConfiguration()
        self.remote_config.update_logging_path(self.results_path)
        self.log.debug(f"Updated dask logging filepath to {self.remote_config.dask_logging_filename}")

        if self.distributed:
            self.log.info("TimeEval is running in distributed environment, setting up remoting ...")
            self.remote = Remote(disable_progress_bar=self.disable_progress_bar, remote_config=self.remote_config,
                                 resource_constraints=limits)
            self.results["future_result"] = np.nan

            self.log.info("... registering signal handlers ...")
            orig_handler: Callable[[int, Optional[FrameType]], Any] = signal.getsignal(signal.SIGINT)  # type: ignore

            def sigint_handler(sig: int, frame: Optional[FrameType] = None) -> Any:
                self.log.warning(f"SIGINT ({sig}) received, shutting down cluster. Please look for dangling Docker "
                                 "containers on all worker nodes (we do not remove them when terminating "
                                 "ungracefully).")
                self.remote.close()
                return orig_handler(sig, frame)
            signal.signal(signal.SIGINT, sigint_handler)

            self.log.info("... remoting setup done.")

    def _run(self) -> None:
        desc = "Submitting evaluation tasks" if self.distributed else "Evaluating"
        for exp in tqdm.tqdm(self.exps, desc=desc, disable=self.disable_progress_bar):
            try:
                future_result: Optional[Future] = None
                result: Optional[Dict] = None

                if exp.algorithm.training_type in [TrainingType.SUPERVISED, TrainingType.SEMI_SUPERVISED]:
                    if exp.resolved_train_dataset_path is None:
                        # Intentionally raise KeyError here if no training dataset is specified.
                        # The Error will be caught by the except clause below.
                        raise KeyError("Path to training dataset not found!")

                    # This check is not necessary for unsupervised algorithms, because they can be executed on all
                    # datasets.
                    if exp.algorithm.training_type != exp.dataset.training_type:
                        raise ValueError(f"Dataset training type ({exp.dataset.training_type}) incompatible to "
                                         f"algorithm training type ({exp.algorithm.training_type})!")

                if (exp.algorithm.input_dimensionality == InputDimensionality.UNIVARIATE and
                        exp.dataset.input_dimensionality == InputDimensionality.MULTIVARIATE):
                    raise ValueError(f"Dataset input dimensionality ({exp.dataset.input_dimensionality}) incompatible "
                                     f"to algorithm input dimensionality ({exp.algorithm.input_dimensionality})!")

                if self.distributed:
                    future_result = self.remote.add_task(exp.evaluate, key=exp.name)
                else:
                    result = exp.evaluate()
                self._record_results(exp, result=result, future_result=future_result)

            except DockerTimeoutError as e:
                self.log.exception(f"Evaluation of {exp.algorithm.name} on the dataset {exp.dataset} timed out.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.TIMEOUT, error_message=repr(e))
            except Exception as e:
                self.log.exception(f"Exception occurred during the evaluation of {exp.algorithm.name} on the dataset {exp.dataset}.")
                result = {m: np.nan for m in self.metric_names}
                self._record_results(exp, result=result, status=Status.ERROR, error_message=repr(e))

    def _record_results(self,
                        exp: Experiment,
                        result: Optional[Dict] = None,
                        future_result: Optional[Future] = None,
                        status: Status = Status.OK,
                        error_message: Optional[str] = None) -> None:
        new_row = {
            "algorithm": exp.algorithm.name,
            "collection": exp.dataset_collection,
            "dataset": exp.dataset_name,
            "algo_training_type": exp.algorithm.training_type.name,
            "algo_input_dimensionality": exp.algorithm.input_dimensionality.name,
            "dataset_training_type": exp.dataset.training_type.name,
            "dataset_input_dimensionality": exp.dataset.input_dimensionality.name,
            "status": status,
            "error_message": error_message,
            "repetition": exp.repetition,
            "hyper_params": dumps_params(exp.params),
            "hyper_params_id": exp.params_id
        }
        if result is not None and future_result is None:
            new_row.update(result)
        elif result is None and future_result is not None:
            new_row.update({"future_result": future_result})
        self.results = self.results.append(new_row, ignore_index=True)
        self.results.replace(to_replace=[None], value=np.nan, inplace=True)

    def _resolve_future_results(self) -> None:
        self.remote.fetch_results()

        result_keys = self.metric_names + Times.result_keys()
        status_keys = ["status", "error_message"]
        keys = result_keys + status_keys

        def get_future_result(f: Future) -> Tuple[Any, ...]:
            try:
                r = f.result()
                return tuple(r.get(k, None) for k in result_keys) + (Status.OK, None)
            except DockerTimeoutError as e:
                self.log.exception(f"Exception {repr(e)} occurred remotely.")
                status = Status.TIMEOUT
                error_message = repr(e)
            except Exception as e:
                self.log.exception(f"Exception {repr(e)} occurred remotely.")
                status = Status.ERROR
                error_message = repr(e)

            return tuple(np.nan for _ in result_keys) + (status, error_message)

        self.results[keys] = self.results["future_result"].apply(get_future_result).tolist()
        self.results = self.results.drop(['future_result'], axis=1)

    def get_results(self, aggregated: bool = True, short: bool = True) -> pd.DataFrame:
        """Return the (aggregated) evaluation results of a previous evaluation run.

        The results are returned in a Pandas :obj:`~pandas.DataFrame` and contain the mean runtime and metrics of the
        algorithms for each dataset.
        You can tweak the output using the parameters.

        .. note::
            Must be called after :func:`~timeeval.TimeEval.run`, otherwise the returned DataFrame is empty.

        Parameters
        ----------
        aggregated : bool
            If ``True``, returns the aggregated results (controled by parameter ``short``), otherwise all collected
            information is returned.
        short : bool
            This parameter is used only in aggregation mode and controls the aggregation level and functions.
            If ``True``, the aggregation is over algorithms and datasets, and the mean of the metrics, training time, and
            execution time is returned.
            If ``False``, the aggregation is over algorithms, datasets, and parameter combinations, and the mean and
            standard deviation of all runtime measurements and metrics are computed.

        Returns
        -------
        :obj:`~pandas.DataFrame` containing the evaluation results.
        """
        if not aggregated:
            return self.results

        df = self.results

        if Status.ERROR in df.status.unique() or Status.TIMEOUT in df.status.unique():
            self.log.warning("The results contain errors which are filtered out for the final aggregation. "
                             "To see all results, call .get_results(aggregated=False)")
            df = df[df.status == Status.OK]

        if short:
            time_names = ["train_main_time", "execute_main_time"]
            group_names = ["algorithm", "collection", "dataset"]
        else:
            time_names = Times.result_keys()
            group_names = ["algorithm", "collection", "dataset", "hyper_params_id"]
        keys = [key for key in self.metric_names + time_names if key in df.columns]
        grouped_results = df.groupby(group_names)
        results: pd.DataFrame = grouped_results[keys].mean()

        if short:
            results = results.rename(columns=dict([(k, f"{k}_mean") for k in keys]))
        else:
            std_results = grouped_results.std()[keys]
            results = results.join(std_results, lsuffix="_mean", rsuffix="_std")
        results["repetitions"] = grouped_results["repetition"].count()
        return results

    def save_results(self, results_path: Optional[Path] = None) -> None:
        """Store the evaluation results to a CSV-file in the provided `results_path`.
        This method is automatically executed by TimeEval at the end of an evaluation run when calling
        :func:`~timeeval.TimeEval.run`.

        Parameters
        ----------
        results_path: Optional[Path]
            Path, where the results should be stored at.
            If it is not supplied, the results path of the current TimeEval run (:attr:`timeeval.TimeEval.results_path`)
            is used.
        """
        path = results_path or (self.results_path / RESULTS_CSV)
        self.results.to_csv(path.resolve(), index=False)

    @staticmethod
    def rsync_results_from(results_path: Path, hosts: List[str], disable_progress_bar: bool = False, n_jobs: int = -1) -> None:
        """Fetches evaluation results of an independent TimeEval run from remote machines merging the temporary data
        and results together on the local host.

        Parameters
        ----------
        results_path : Path
            Path to the evaluation results.
            Must be the same for all hosts.
        hosts : List[str]
            List of hostnames or IP addresses that took part in the evaluation run.
        disable_progress_bar : bool
            If a progress bar should be displayed or not.
        n_jobs : int
            Number of parallel processes used to fetch the results.
            The parallelism is limited by the number of external hosts and the maximum number of available CPU cores.
        """
        results_path = results_path.resolve()
        hostname = socket.gethostname()
        excluded_aliases = [
            hostname,
            socket.gethostbyname(hostname),
            "localhost",
            socket.gethostbyname("localhost")
        ]
        jobs = [
            delayed(subprocess.call)(["rsync", "-a", f"{host}:{results_path}/", f"{results_path}"])
            for host in hosts
            if host not in excluded_aliases
        ]
        with tqdm_joblib(tqdm.tqdm(hosts, desc="Collecting results", disable=disable_progress_bar, total=len(jobs))):
            Parallel(n_jobs)(jobs)

    def rsync_results(self) -> None:
        """Fetches the evaluation results of the current evaluation run from all remote machines merging the temporary
        data and results together on the local host.
        This method is automatically executed by TimeEval at the end of an evaluation run started by calling
        :func:`~timeeval.TimeEval.run`.

        See Also
        --------
        timeeval.TimeEval.rsync_results_from
        """
        TimeEval.rsync_results_from(
            self.results_path,
            self.remote_config.worker_hosts,
            self.disable_progress_bar,
            self.n_jobs
        )

    def _prepare(self) -> None:
        n = len(self.exps)
        self.log.debug(f"Running {n} algorithm prepare steps")
        for algorithm in self.exps.algorithms:
            algorithm.prepare()
        self.log.debug(f"Creating {n} result directories")
        for exp in self.exps:
            exp.results_path.mkdir(parents=True, exist_ok=True)

    def _finalize(self) -> None:
        self.log.debug(f"Running {len(self.exps)} algorithm finalize steps")
        for algorithm in self.exps.algorithms:
            algorithm.finalize()

    def _distributed_prepare(self) -> None:
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            prepare_fn = algorithm.prepare_fn()
            if prepare_fn:
                tasks.append((prepare_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm prepare steps")

        def mkdirs(dirs: List[Path]) -> None:
            for d in dirs:
                d.mkdir(parents=True, exist_ok=True)

        dir_list = [exp.results_path for exp in self.exps]
        tasks.append((mkdirs, [dir_list], {}))
        self.log.debug(f"Collected {len(dir_list)} directories to create on remote nodes")
        self.remote.run_on_all_hosts(tasks, msg="Preparing")

    def _distributed_finalize(self) -> None:
        tasks: List[Tuple[Callable, List, Dict]] = []
        for algorithm in self.exps.algorithms:
            finalize_fn = algorithm.finalize_fn()
            if finalize_fn:
                tasks.append((finalize_fn, [], {}))
        self.log.debug(f"Collected {len(tasks)} algorithm finalize steps")
        self.log.info("Running finalize steps on remote hosts")
        self.remote.run_on_all_hosts(tasks, msg="Finalizing")
        self.log.info("Closing remote")
        self.remote.close()
        self.log.info("Syncing results")
        self.rsync_results()

    def run(self) -> None:
        """Starts the configured evaluation run.

        Each TimeEval run consists of a number of experiments that are executed independently of each other.
        There are three phases: PREPARE, EVALUATION, FINALIZE.

        1. _PREPARE_ phase: In the first phase, the execution environment is prepared, the result folder is created, and
           algorithm adapter-dependent preparation steps, such as pulling Docker images for the
           :class:`~timeeval.adapters.docker.DockerAdapter`, are executed.
        2. _EVALUATION_ phase: In the evaluation phase, the experiments are executed and the results are recorded and
           stored to disk.
        3. _FINALIZE_ phase: In the last phase, the execution environment is cleaned up, and algorithm adapter-dependent
           finalization steps, such as removing the temporary Docker containers for the
           :class:`~timeeval.adapters.docker.DockerAdapter`, are executed.

        This method executes all three phases after each other and returns after they are finished.
        You can access the evaluation results either using :func:`~timeeval.TimeEval.get_results` programmatically or
        in the results folder from the file system.
        """
        print("Running PREPARE phase")
        self.log.info("Running PREPARE phase")
        t0 = time()
        if self.distributed:
            self._distributed_prepare()
        else:
            self._prepare()

        print("Running EVALUATION phase")
        self.log.info("Running EVALUATION phase")
        self._run()
        if self.distributed:
            self._resolve_future_results()
        self.save_results()

        print("Running FINALIZE phase")
        self.log.info("Running FINALIZE phase")
        if self.distributed:
            self._distributed_finalize()
        else:
            self._finalize()
        t1 = time()

        msg = f"""FINALIZE phase done.
          Stored results at {self.results_path / RESULTS_CSV}.
          Overall runtime of this TimeEval run: {t1 - t0} seconds
        """
        print(msg)
        self.log.info(msg)
