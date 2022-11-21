#!/usr/bin/env python3
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, RemoteConfiguration
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS
from timeeval.params import FixedParameters
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
from timeeval_experiments.algorithms import kmeans, subsequence_knn, subsequence_lof, stomp, grammarviz3, dwt_mlead, \
    torsk, subsequence_if


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
    dm = MultiDatasetManager(
        [],
        # custom_datasets_file=Path("../../data/custom/univariate-SVDB/custom_datasets.json")
        custom_datasets_file=Path("/home/projects/akita/data/custom/univariate-SVDB/custom_datasets.json")
    )
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = dm.select(collection="custom")
    algorithms = [
        subsequence_lof(),
        subsequence_if(),
        stomp(),
        grammarviz3(),
        dwt_mlead(),
        kmeans(),
    ]
    configurator.configure(algorithms, perform_search=False)
    algorithms.append(subsequence_knn(FixedParameters({
        "window_size": "heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)",
        "distance_metric_order": 2,
        "leaf_size": 20,
        "method": "largest",
        "n_neighbors": 50,
        "radius": 1.0,
        "n_jobs": 1,
        "random_state": 42,
    })))
    algorithms.append(torsk(FixedParameters({
        "context_window_size": 10,
        "density": 0.01,
        "imed_loss": False,
        "input_map_scale": 0.125,
        "input_map_size": 100,
        "prediction_window_size": 5,
        "reservoir_representation": "sparse",
        "scoring_large_window_size": 100,
        "scoring_small_window_size": 10,
        "spectral_radius": 2.0,
        "tikhonov_beta": None,
        "train_method": "pinv_svd",
        "train_window_size": 100,
        "transient_window_size": 20
    })))

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
                        metrics=[RangePrAUC(buffer_size=100), RangeRocAUC(buffer_size=100), RangePrVUS(),
                                 RangeRocVUS()],
                        )

    # copy parameter configuration file to results folder
    timeeval.results_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configurator.config_path, timeeval.results_path)

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
