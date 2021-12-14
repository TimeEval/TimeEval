#!/usr/bin/env python3
import logging
import random
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, TrainingType, DatasetManager
from timeeval.constants import HPI_CLUSTER
from timeeval.params import FullParameterGrid
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval.utils.metrics import Metric
from timeeval_experiments.algorithms import *


# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.INFO,
    force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)

random.seed(42)
np.random.rand(42)
MAX_CONTAMINATION = 0.1
MIN_ANOMALIES = 1


def main():
    # from pathlib import Path
    # root_data_path = Path("../../data")
    # dm = DatasetManager(
    #     root_data_path / "test-cases"
    # )
    dm = DatasetManager(
        HPI_CLUSTER.akita_correlation_anomalies_path
    )

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
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
        for param in algo.param_grid:
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
        train_fails_on_timeout=False,
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
