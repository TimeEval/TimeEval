#!/usr/bin/env python3
import logging
import random
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality
from timeeval.constants import HPI_CLUSTER
from timeeval.params import FixedParameters
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval.utils.metrics import Metric
# Setup logging
from timeeval_experiments.algorithms import grammarviz3_multi, multinorma, kmeans


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
        HPI_CLUSTER.akita_benchmark_path,
        HPI_CLUSTER.akita_test_case_path,
        HPI_CLUSTER.akita_correlation_anomalies_path
    ])
    # dm = MultiDatasetManager([
    #     "../../data/benchmark-data/data-processed",
    #     "../../data/test-cases",
    #     "../../data/correlation-anomalies"
    # ])

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
    datasets += dm.select(
        max_contamination=MAX_CONTAMINATION,
        min_anomalies=MIN_ANOMALIES,
        input_dimensionality=InputDimensionality.MULTIVARIATE,
    )
    # exclude too large datasets
    # and exclude GutenTAG dataset, because they contain semi-, supervised, and unsupervised datasets that are the same
    datasets = [(c, d) for c, d in datasets if c not in ["Exathlon", "IOPS", "LTDB", "Kitsune", "GutenTAG"]]
    # add the multivariate GutenTAG datasets, but just use the unsupervised ones
    datasets += dm.select(
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE
    )

    algorithms = [
        kmeans(),
        grammarviz3_multi(FixedParameters({
            "alphabet_size": 7,
            "paa_transform_size": 5,
            "n_discords": 100,
            "anomaly_window_size": "heuristic:PeriodSizeHeuristic(factor=1.5, fb_value=150)",
            "multi_strategy": 1,
            "numerosity_reduction": True,
            "output_mode": 0
        })),
        multinorma(FixedParameters({
            "motif_detection": "mixed",
            "sum_dims": False,
            "normalize_join": True,
            "join_combine_method": 1,
            "anomaly_window_size": "heuristic:AnomalyLengthHeuristic(agg_type='max')",
            "normal_model_percentage": 0.5,
            "max_motifs": 4096,
            "random_state": 42
        }))
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
        task_memory_limit=3 * GB,
        use_preliminary_scores_on_execute_timeout=True,
        train_timeout=Duration("2 hours"),
        execute_timeout=Duration("2 hours"),
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
