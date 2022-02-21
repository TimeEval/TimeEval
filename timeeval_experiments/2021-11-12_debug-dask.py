#!/usr/bin/env python3
import logging
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, Datasets, TrainingType
from timeeval.constants import HPI_CLUSTER
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval.utils.metrics import Metric
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines


# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.DEBUG,
    # force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)

random.seed(42)
np.random.rand(42)


def main():
    dm = Datasets(HPI_CLUSTER.akita_benchmark_path, create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
    datasets += dm.select(collection_name="KDD-TSAD", dataset_name="218_UCR_Anomaly_STAFFIIIDatabase")
    datasets += dm.select(collection_name="KDD-TSAD", dataset_name="237_UCR_Anomaly_mit14157longtermecg")
    datasets += dm.select(collection_name="KDD-TSAD", dataset_name="177_UCR_Anomaly_insectEPG5")
    datasets += dm.select(collection_name="NAB", dataset_name="art_daily_no_noise")
    datasets += dm.select(collection_name="WebscopeS5", dataset_name="A2Benchmark-82")
    datasets += dm.select(collection_name="WebscopeS5", dataset_name="A2Benchmark-73")
    datasets += dm.select(dataset_name="qtdbSel100MLII")
    datasets += dm.select(dataset_name="G-1")
    datasets += random.sample(dm.select(collection_name="Exathlon", train_type=TrainingType.SUPERVISED.value), 2)
    datasets += random.sample(dm.select(collection_name="Exathlon", train_type=TrainingType.SEMI_SUPERVISED.value), 2)
    print(f"Selecting {len(datasets)} datasets")

    algorithms = [
        cblof(),
        left_stampi(),
        grammarviz3(),
        pst(),
        robust_pca(),
        s_h_esd(),
        stamp(),
        stomp(),
        baseline_normal(),  # Docker-based
        Baselines.normal()  # python-fn-based
    ]
    print(f"Selecting {len(algorithms)} algorithms")

    print("Configuring algorithms...")
    configurator.configure(algorithms, perform_search=False)

    print("\nDatasets:")
    print("=====================================================================================")
    for collection in np.unique([c for (c, d) in datasets]):
        print(collection)
        cds = sorted([d for (c, d) in datasets if c == collection])
        for cd in cds:
            print(f"  {cd}")
    print("=====================================================================================\n\n")

    print("\nParameter configurations:")
    print("=====================================================================================")
    for algo in algorithms:
        print(algo.name)
        for param in algo.param_config:
            print(f"  {param}")
    print("=====================================================================================\n\n")
    sys.stdout.flush()

    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=HPI_CLUSTER.nodes,
        dask_logging_file_level="DEBUG",
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=3*GB,
        use_preliminary_model_on_train_timeout=True,
        train_timeout=Duration("2 hours"),
        execute_timeout=Duration("2 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        force_dimensionality_match=False,
                        force_training_type_match=False,
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC, Metric.AVERAGE_PRECISION],
                        )

    # copy parameter configuration file to results folder
    shutil.copy2(configurator.config_path, timeeval.results_path)

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
