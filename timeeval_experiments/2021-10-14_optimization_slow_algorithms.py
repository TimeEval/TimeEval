#!/usr/bin/env python3
import logging
import random
import sys

import numpy as np
from durations import Duration

from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER
from timeeval.params import IndependentParameterGrid, FullParameterGrid
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints
from timeeval.utils.metrics import Metric
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
from timeeval_experiments.algorithms import *


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
    dm = Datasets(HPI_CLUSTER.akita_test_case_path, create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets = dm.select()
    datasets = [(collection, name) for (collection, name) in datasets
                if not name.startswith("cbf-") or not name.startswith("rw-")]
    all_datasets = datasets
    datasets = []
    datasets += random.sample([(c, d) for (c, d) in all_datasets if d.startswith("sinus")], 10)
    datasets += random.sample([(c, d) for (c, d) in all_datasets if d.startswith("ecg")], 10)
    datasets += random.sample([(c, d) for (c, d) in all_datasets if d.startswith("poly")], 10)
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        # arima(),
        # deepnap(timeout=Duration("4 hours")),
        # pst()
    ]

    print("Configuring algorithms...")
    configurator.configure(algorithms,
                           ignore_shared=False,  # use already optimized shared parameters
                           perform_search=True,  # but perform search over the optimized parameter search spaces
                           assume_parameter_independence=True
                           )
    algorithms.append(
        arima(params=IndependentParameterGrid({
            "distance_metric": ["euclidean", "mahalanobis", "garch", "ssa", "fourier", "dtw", "edrs", "twed"]
        }, default_params={
            "window_size": "heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)",
            "max_lag": "heuristic:RelativeDatasetSizeHeuristic(factor=0.1)",
            "differencing_degree": 1,
            "random_state": 42,
        })),
    )
    # algorithms.append(
    #     numenta_htm(params=IndependentParameterGrid({
    #         "alpha": [0.2, 0.5, 0.8],
    #         "globalDecay": [0, 0.1, 0.5],
    #         "encoding_output_width": [25, 50, 75],
    #         "encoding_input_width": [15, 21, 30],
    #         "columnCount": [1024, 2048, 4096],
    #         "cellsPerColumn": [16, 32, 64],
    #         "autoDetectWaitRecords": [25, 50, 75],
    #         "activationThreshold": [6, 12, 24],
    #         "inputWidth": [1024, 2048, 4096],
    #         "initialPerm": [0.15, 0.21, 0.3],
    #         "maxAge": [0, 5, 10],
    #         "synPermConnected": [0.05, 0.1, 0.2],
    #         "synPermInactiveDec": [0.001, 0.005, 0.01],
    #         "synPermActiveInc": [0.05, 0.1, 0.2],
    #         "maxSegmentsPerCell": [64, 128, 256],
    #         "potentialPct": [0.1, 0.5, 0.9],
    #         "permanenceInc": [0.05, 0.1, 0.2],
    #         "permanenceDec": [0.05, 0.1, 0.2],
    #         "pamLength": [1, 3, 5],
    #         "numActiveColumnsPerInhArea": [30, 40, 50],
    #         "newSynapseCount": [15, 20, 30],
    #         "minThreshold": [6, 9, 12],
    #         "maxSynapsesPerSegment": [16, 32, 64]
    #     }))
    # )

    algorithms.append(
        random_black_forest(params=FullParameterGrid({
            "n_trees": [10, 100, 200],
            "n_estimators": [10, 100, 200],
            "bootstrap": [True, False]
        }))
    )

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
        use_preliminary_model_on_train_timeout=True,
        train_timeout=Duration("2 hours"),
        execute_timeout=Duration("2 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        force_training_type_match=True,
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC, Metric.RANGE_PR_AUC, Metric.AVERAGE_PRECISION],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
