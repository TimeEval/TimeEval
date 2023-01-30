#!/usr/bin/env python3
import logging
import sys
import random
from typing import List, Tuple

import numpy as np
from durations import Duration
from optuna import distributions

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality, RemoteConfiguration
from timeeval.constants import HPI_CLUSTER
from timeeval.integration.optuna import OptunaConfiguration, OptunaStudyConfiguration
from timeeval.metrics import RangePrAUC, RangeRocAUC
from timeeval.params import BayesianParameterSearch
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval_experiments.algorithms import subsequence_knn, subsequence_lof, subsequence_if, stomp, kmeans, \
    dwt_mlead, torsk


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

def main():
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK],
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.UNIVARIATE_ANOMALY_TEST_CASES],
    ])

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = dm.select(collection="Dodgers")
    datasets += random.sample(dm.select(
        input_dimensionality=InputDimensionality.UNIVARIATE,
        training_type=TrainingType.UNSUPERVISED,
        collection="univariate-anomaly-test-cases",
    ), 3)
    datasets += dm.select(dataset="sine-difflen-3-frequency")
    datasets += dm.select(dataset="sine-difflen-3-variance")
    datasets += random.sample(dm.select(collection="KDD-TSAD", max_contamination=0.1), 2)
    datasets += random.sample(dm.select(collection="NASA-MSL", max_contamination=0.1), 2)

    study_config = OptunaStudyConfiguration(n_trials=300, metric=RangePrAUC(buffer_size=100))
    algorithms = [
        dwt_mlead(BayesianParameterSearch(study_config, include_default_params=True, params={
            "quantile_epsilon": distributions.FloatDistribution(0.001, 0.5),
            "start_level": distributions.IntDistribution(1, 10),
        })),
        kmeans(BayesianParameterSearch(study_config, include_default_params=True, params={
            "anomaly_window_size": distributions.IntDistribution(10, 2000),
            "n_clusters": distributions.IntDistribution(2, 100),
        })),
        stomp(BayesianParameterSearch(study_config, include_default_params=True, params={
            "anomaly_window_size": distributions.IntDistribution(10, 2000),
            "exclusion_zone": distributions.FloatDistribution(0, 0.75),
        })),
        subsequence_if(BayesianParameterSearch(study_config, include_default_params=True, params={
            "window_size": distributions.IntDistribution(10, 2000),
            "n_trees": distributions.IntDistribution(10, 1000),
            "bootstrap": distributions.CategoricalDistribution([True, False]),
            "max_features": distributions.FloatDistribution(0.01, 1.0),
            "max_samples": distributions.FloatDistribution(0.01, 1.0),
        })),
        subsequence_lof(BayesianParameterSearch(study_config, include_default_params=True, params={
            "window_size": distributions.IntDistribution(10, 2000),
            "distance_metric_order": distributions.IntDistribution(1, 4),
            "leaf_size": distributions.IntDistribution(10, 100),
            "n_neighbors": distributions.IntDistribution(1, 100),
        })),
        subsequence_knn(BayesianParameterSearch(study_config, include_default_params=True, params={
            "window_size": distributions.IntDistribution(10, 2000),
            "distance_metric_order": distributions.IntDistribution(1, 4),
            "leaf_size": distributions.IntDistribution(10, 100),
            "n_neighbors": distributions.IntDistribution(1, 100),
            "method": distributions.CategoricalDistribution(["largest", "mean", "median"]),
            "radius": distributions.FloatDistribution(0.1, 5.0),
        })),
        torsk(BayesianParameterSearch(study_config.copy(n_trials=900), include_default_params=True, params={
            "context_window_size": distributions.IntDistribution(10, 2000),
            "density": distributions.FloatDistribution(0.0001, 1, log=True),
            "imed_loss": distributions.CategoricalDistribution([True, False]),
            "input_map_scale": distributions.FloatDistribution(0.001, 1.0),
            "input_map_size": distributions.IntDistribution(10, 2000),
            "prediction_window_size": distributions.IntDistribution(10, 1000),
            "scoring_large_window_size": distributions.IntDistribution(50, 2000),
            "scoring_small_window_size": distributions.IntDistribution(10, 1000),
            "spectral_radius": distributions.FloatDistribution(1.0, 10.0),
            "train_window_size": distributions.IntDistribution(10, 1000),
            "transient_window_size": distributions.IntDistribution(10, 1000),
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
        print(f"{algo.name}: {len(algo.param_config)}")
    print("=====================================================================================\n\n")
    print(f"Datasets: {len(datasets)}")
    print(f"Algorithms: {len(algorithms)}")
    sys.stdout.flush()

    optuna_config = OptunaConfiguration(
        default_storage="postgresql",
        dashboard=True,
        remove_managed_containers=False
    )
    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=list(set(HPI_CLUSTER.nodes) - {HPI_CLUSTER.odin02, HPI_CLUSTER.odin03}),
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=6 * GB,
        execute_timeout=Duration("4 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[RangeRocAUC(buffer_size=100), RangePrAUC(buffer_size=100)],
                        module_configs={"optuna": optuna_config},
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
