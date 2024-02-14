#!/usr/bin/env python3
import logging
import sys
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, TrainingType, InputDimensionality, RemoteConfiguration
from timeeval.algorithm import Algorithm
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import RocAUC, PrAUC, RangePrAUC, RangeRocAUC, RangePrVUS, RangeRocVUS
from timeeval.params.base import FixedParameters
from timeeval.resource_constraints import ResourceConstraints, GB
from timeeval_experiments.algorithms import kmeans, torsk, random_black_forest, lstm_ad, dbstream, normalizing_flows
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


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


_bad_parameters: Dict[str, Dict[str, Any]] = {
    "context": {
        "name": "context",
        "type": "Int",
        "defaultValue": 10,
        "description": "Context size in both directions."
    }
}


def bad_timeeval(name: str, params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name=name,
        main=DockerAdapter(
            image_name="bad",
            tag="last",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_bad_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )


def define_datasets() -> Tuple[List[Tuple[str, str]], MultiDatasetManager]:
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK]
    ])

    datasets = dm.select(
        input_dimensionality=InputDimensionality.UNIVARIATE,
        max_contamination=MAX_CONTAMINATION,
        min_anomalies=MIN_ANOMALIES
    )

    return datasets, dm


def define_algorithms() -> List[Algorithm]:
    return [
        bad_timeeval(
            name="BAD",
            params=FixedParameters({
                "context": "heuristic:AnomalyLengthHeuristic(agg_type='max')",
            }),
            skip_pull=True
        ),
        bad_timeeval(
            name="BAD-10",
            params=FixedParameters({
                "context": 10,
            }),
            skip_pull=True
        )
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
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=5 * GB,
        train_timeout=Duration("1 hour"),
        execute_timeout=Duration("1 hour"),
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
                        ],
                )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
