#!/usr/bin/env python3
import logging
import random
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality
from timeeval.constants import HPI_CLUSTER
from timeeval.params import FixedParameters, IndependentParameterGrid
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval.utils.metrics import Metric
from timeeval_experiments.algorithms import grammarviz3_multi, multinorma, kmeans, multi_subsequence_lof, torsk, \
    subsequence_knn, mstamp


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
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_test_case_path,
        HPI_CLUSTER.akita_benchmark_path
    ])

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = dm.select(
        input_dimensionality=InputDimensionality.UNIVARIATE,
        training_type=TrainingType.UNSUPERVISED,
        max_contamination=MAX_CONTAMINATION,
        min_anomalies=MIN_ANOMALIES
    )
    algorithms = [
        subsequence_knn(FixedParameters({
            "window_size": "heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)",
            "distance_metric_order": 2,
            "leaf_size": 20,
            "method": "largest",
            "n_neighbors": 50,
            "radius": 1.0,
            "n_jobs": 1,
            "random_state": 42,
        })),
    ]

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
    print(f"Datasets: {len(datasets)}")
    print(f"Algorithms: {len(algorithms)}")
    sys.stdout.flush()

    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=HPI_CLUSTER.nodes
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=6 * GB,
        use_preliminary_scores_on_execute_timeout=True,
        execute_timeout=Duration("4 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
