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
from timeeval.params import IndependentParameterGrid, FixedParameters
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


def multi_grammarviz_algorithms():
    docker_image_name = "sopedu:5000/akita/grammarviz3-multi"
    docker_image_tag = "280553e5"

    default_params = {
        "alphabet_size": 6,
        "paa_transform_size": 5,
        "anomaly_window_size": "heuristic:PeriodSizeHeuristic(factor=1.5, fb_value=150)"
    }
    explorative_params = {
        "alphabet_size": [4, 5, 7],
        "paa_transform_size": [3, 4, 6],
        "anomaly_window_size": "heuristic:PeriodSizeHeuristic(factor=1.5, fb_value=150)"
    }
    extended_explorative_params = {
        "alphabet_size": [4, 5, 7, 8],
        "paa_transform_size": [4, 6, 7, 8, 9],
        "anomaly_window_size": ["heuristic:PeriodSizeHeuristic(factor=1.5, fb_value=150)", 150]
    }

    algorithms = [
        # density algorithms
        Algorithm(
            name="grammarviz-mv-density-sep",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(extended_explorative_params,
                                                  {**default_params, "output_mode": 0, "multi_strategy": 2})
        ),
        Algorithm(
            name="grammarviz-mv-density-cluster",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(extended_explorative_params,
                                                  {**default_params, "output_mode": 0, "multi_strategy": 1})
        ),
        Algorithm(
            name="grammarviz-mv-density-merge",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(extended_explorative_params,
                                                  {**default_params, "output_mode": 0, "multi_strategy": 0})
        ),
        # discord algorithms
        Algorithm(
            name="grammarviz-mv-discord-sep",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(explorative_params,
                                                  {**default_params, "output_mode": 1, "multi_strategy": 2})
        ),
        Algorithm(
            name="grammarviz-mv-discord-cluster",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(explorative_params,
                                                  {**default_params, "output_mode": 1, "multi_strategy": 1})
        ),
        Algorithm(
            name="grammarviz-mv-discord-merge",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(extended_explorative_params,
                                                  {**default_params, "output_mode": 1, "multi_strategy": 0})
        ),
        # full algorithms
        Algorithm(
            name="grammarviz-mv-full-sep",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(explorative_params,
                                                  {**default_params, "output_mode": 2, "multi_strategy": 2})
        ),
        Algorithm(
            name="grammarviz-mv-full-cluster",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(explorative_params,
                                                  {**default_params, "output_mode": 2, "multi_strategy": 1})
        ),
        Algorithm(
            name="grammarviz-mv-full-merge",
            main=DockerAdapter(image_name=docker_image_name, tag=docker_image_tag),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=IndependentParameterGrid(extended_explorative_params,
                                                  {**default_params, "output_mode": 2, "multi_strategy": 0})
        ),
    ]
    return algorithms


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
            name="DPIE",
            main=DockerAdapter(image_name="sopedu:5000/akita/dpie", tag="9e9abd94"),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=FixedParameters({"dataset_id": "heuristic:DatasetIdHeuristic()"})
        ),
        Algorithm(
            name="MultiNormA",
            main=DockerAdapter(image_name="sopedu:5000/akita/multinorma", tag="3545a123"),
            data_as_file=True,
            training_type=TrainingType.UNSUPERVISED,
            input_dimensionality=InputDimensionality.MULTIVARIATE,
            param_config=FixedParameters({
                "anomaly_window_size": "heuristic:AnomalyLengthHeuristic(agg_type='max')",
                "normal_model_percentage": 0.5,
                "random_state": 42
            })
        )
    ]
    algorithms.extend(multi_grammarviz_algorithms())

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
