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
from timeeval.utils.window import ReverseWindowing
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


"""
############## s2gpp
"""

def post_s2gpp(scores: np.ndarray, args: dict) -> np.ndarray:
    pattern_length = args.get("hyper_params", {}).get("pattern-length", 50)
    query_length = args.get("hyper_params", {}).get("query-length", 75)
    size = pattern_length + query_length
    return ReverseWindowing(window_size=size).fit_transform(scores)


_s2gpp_parameters: Dict[str, Dict[str, Any]] = {
    "pattern-length": {
        "name": "pattern-length",
        "type": "Int",
        "defaultValue": 50,
        "description": "Size of the sliding window, independent of anomaly length, but should in the best case be larger."
    },
    "latent": {
        "name": "latent",
        "type": "Int",
        "defaultValue": 16,
        "description": "Size of latent embedding space. This space is the input for the PCA calculation afterwards."
    },
    "rate": {
        "name": "rate",
        "type": "Int",
        "defaultValue": 100,
        "description": "Number of angles used to extract pattern nodes. A higher value will lead to high precision, but at the cost of increased computation time."
    },
    "threads": {
        "name": "threads",
        "type": "Int",
        "defaultValue": 1,
        "description": "Number of helper threads started besides the main thread. (min=1)"
    },
    "query-length": {
        "name": "query-length",
        "type": "Int",
        "defaultValue": 75,
        "description": "Size of the sliding windows used to find anomalies (query subsequences). query-length must be >= pattern-length!"
    },
    "clustering": {
        "name": "clustering",
        "type": "String",
        "defaultValue": "meanshift",
        "description": "Determines which clustering algorithm to use. Possible choices are: `meanshift` or `kde`."
    }
}


def s2gpp_timeeval(name: str, params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name=name,
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/s2gpp",
            tag="1.0.2",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=post_s2gpp,
        param_schema=_s2gpp_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )


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
        kmeans(skip_pull=True),
        torsk(skip_pull=True),
        random_black_forest(skip_pull=True),
        lstm_ad(skip_pull=True),
        dbstream(skip_pull=True),
        normalizing_flows(skip_pull=True),
        s2gpp_timeeval(
            name="S2G++",
            params=FixedParameters({
                "rate": 100,
                "pattern-length": "heuristic:AnomalyLengthHeuristic(agg_type='max')",
                "latent": "heuristic:ParameterDependenceHeuristic(source_parameter='pattern-length', factor=1./3.)",
                "query-length": "heuristic:ParameterDependenceHeuristic(source_parameter='pattern-length', factor=1.0)",
                "threads": 20,
                "clustering": "kde",
                "self-correction": ""
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
        tasks_per_host=3,
        task_cpu_limit=1.,
        task_memory_limit=20 * GB,
        train_timeout=Duration("12 hours"),
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
