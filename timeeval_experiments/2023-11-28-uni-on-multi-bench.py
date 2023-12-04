#!/usr/bin/env python3
import logging
import sys
import random
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality, RemoteConfiguration
from timeeval.algorithm import Algorithm
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import RocAUC, PrAUC, RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval_experiments.algorithms import subsequence_lof, dwt_mlead, subsequence_if, norma, series2graph, kmeans, stamp, mstamp
from timeeval.adapters.multivar import MultivarAdapter, AggregationMethod


# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.INFO,
    # force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)

MAX_CONTAMINATION = 0.1
MIN_ANOMALIES = 1

random.seed(42)
np.random.rand(42)


def wrap_multivar(algorithm: Algorithm, aggregation_method: AggregationMethod) -> Algorithm:
    algorithm.name = f"{algorithm.name}_multi({aggregation_method.name})"
    algorithm.training_type = TrainingType.UNSUPERVISED
    algorithm.input_dimensionality = InputDimensionality.MULTIVARIATE
    algorithm.main = MultivarAdapter(algorithm.main, aggregation_method)
    return algorithm


def define_datasets() -> Tuple[List[Tuple[str, str]], MultiDatasetManager]:
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK]
    ])

    datasets = dm.select(
        input_dimensionality=InputDimensionality.MULTIVARIATE,
        max_contamination=MAX_CONTAMINATION,
        min_anomalies=MIN_ANOMALIES
    )

    return datasets, dm


def define_algorithms() -> List[Algorithm]:
    return [
        wrap_multivar(subsequence_lof(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(subsequence_lof(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(subsequence_lof(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(dwt_mlead(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(dwt_mlead(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(dwt_mlead(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(subsequence_if(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(subsequence_if(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(subsequence_if(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(norma(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(norma(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(norma(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(series2graph(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(series2graph(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(series2graph(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(kmeans(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(kmeans(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(kmeans(skip_pull=True), AggregationMethod.SUM_BEFORE),

        wrap_multivar(stamp(skip_pull=True), AggregationMethod.MEAN),
        wrap_multivar(stamp(skip_pull=True), AggregationMethod.MAX),
        wrap_multivar(stamp(skip_pull=True), AggregationMethod.SUM_BEFORE),

        #mstamp(),
    ]


def main():

    algorithms = define_algorithms()
    datasets, dm = define_datasets()

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


    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=list(set(HPI_CLUSTER.nodes) - {HPI_CLUSTER.odin14}),
    )
    limits = ResourceConstraints(
        tasks_per_host=3,
        task_cpu_limit=1.,
        task_memory_limit=20 * GB,
        execute_timeout=Duration("12 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[
                            RocAUC(),
                            PrAUC(),
                            RangeRocAUC(buffer_size=100),
                            RangePrAUC(buffer_size=100),
                            RangePrVUS(),
                            RangeRocVUS()
                        ],
                )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))



if __name__ == "__main__":
    main()
