#!/usr/bin/env python3
import logging
import random
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality, Algorithm
from timeeval.adapters import DockerAdapter
from timeeval.constants import HPI_CLUSTER
from timeeval.params import IndependentParameterGrid, FixedParameters, FullParameterGrid
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval.utils.metrics import Metric


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
    # add the GutenTAG datasets (univariate and multivariate), but just use the unsupervised ones
    datasets += dm.select(
        collection="GutenTAG",
        training_type=TrainingType.UNSUPERVISED
    )

    algorithms = [
        Algorithm(
            name="MultiNormA",
            main=DockerAdapter(image_name="sopedu:5000/akita/multinorma", tag="40e1bc8e"),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=FullParameterGrid({
                "motif_detection": [0, 1],
                "dim_handling": [0, 1],
                "normalize_join": [0, 1],
                "anomaly_window_size": ["heuristic:AnomalyLengthHeuristic(agg_type='max')"],
                "normal_model_percentage": [0.5],
                "random_state": [42]
            })
        )
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
        use_preliminary_model_on_train_timeout=True,
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
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC, Metric.RANGE_PR_AUC, Metric.AVERAGE_PRECISION],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
