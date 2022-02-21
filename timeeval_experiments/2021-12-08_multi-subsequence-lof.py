#!/usr/bin/env python3
import logging
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration
from timeeval.params import FullParameterGrid

from timeeval import TimeEval, Datasets, TrainingType, InputDimensionality, MultiDatasetManager
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
    level=logging.INFO,
    # force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)

random.seed(42)
np.random.rand(42)
MAX_CONTAMINATION = 0.1
MIN_ANOMALIES = 1


def main():
    # from pathlib import Path
    # root_data_path = Path("../../data")
    # dm = MultiDatasetManager([
    #     root_data_path / "benchmark-data" / "data-processed",
    #     root_data_path / "test-cases",
    #     root_data_path / "correlation-anomalies"
    # ])
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_benchmark_path,
        HPI_CLUSTER.akita_test_case_path,
        HPI_CLUSTER.akita_correlation_anomalies_path
    ])

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
    datasets += dm.select(collection="CalIt2", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="Daphnet", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # no datasets match criteria for Dodgers
    # datasets += dm.select(collection="Dodgers", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # select 4 datasets of large-timeseries collection Exathlon
    datasets += random.sample(dm.select(collection="Exathlon", training_type=TrainingType.SUPERVISED), 2)
    datasets += random.sample(dm.select(collection="Exathlon", training_type=TrainingType.SEMI_SUPERVISED), 2)
    datasets += dm.select(collection="GHL", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="Genesis", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # GutenTAG uses a separate run
    # select 4 datasets of large-timeseries collection IOPS
    datasets += random.sample(dm.select(collection="IOPS", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES), 4)
    # include everything from KDD-TSAD!
    datasets += dm.select(collection="KDD-TSAD")
    datasets += dm.select(collection="Keogh", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # exclude Kitsune completely, bc it's too large!
    # datasets += random.sample(dm.select(collection="Kitsune", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES), 4)
    # exclude LTDB completely, bc it's too large!
    # datasets += random.sample(dm.select(collection="LTDB", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES), 4)
    datasets += dm.select(collection="MGAB", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="MITDB", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="Metro", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # include everything from NAB!
    datasets += dm.select(collection="NAB")
    datasets += dm.select(collection="NASA-MSL", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="NASA-SMAP", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="OPPORTUNITY", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # no datasets match criteria for Occupancy
    datasets += dm.select(collection="Occupancy", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="SMD", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # no datasets match criteria for SSA, bc contamination > 0.14 for all datasets
    # datasets += dm.select(collection="SSA")
    datasets += dm.select(collection="SVDB", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="WebscopeS5", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    # use only unsupervised datasets for the multivariate Subsequence LOF algorithm of the GutenTAG collection
    datasets += dm.select(collection="GutenTAG", training_type=TrainingType.UNSUPERVISED)
    print(f"Selecting {len(datasets)} datasets")

    algorithms = [
        subsequence_lof_multi_sum(FullParameterGrid({
            "window_size": ["heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)"],
            "distance_metric_order": [2],
            "leaf_size": [20],
            "n_neighbors": [50],
        }))
    ]
    print(f"Selecting {len(algorithms)} algorithms")

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
        worker_hosts=HPI_CLUSTER.nodes
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
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC, Metric.RANGE_PR_AUC, Metric.AVERAGE_PRECISION],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
