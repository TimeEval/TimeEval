#!/usr/bin/env python3
import logging
import random
import sys

import numpy as np
from durations import Duration

from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints
from timeeval.utils.metrics import Metric
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
from timeeval_experiments.algorithms import *
from timeeval_experiments.baselines import Baselines


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
    algo_page_size = 5
    algo_page = 5

    dm = Datasets(HPI_CLUSTER.akita_test_case_path, create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets = dm.select()
    datasets = [(collection, name) for (collection, name) in datasets
                if not name.startswith("cbf-") or not name.startswith("rw-")]
    # datasets = random.sample(datasets, 200)
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        dbstream(),
        deepant(),
        donut(),
        dwt_mlead(),
        fft(),
        generic_rf(),
        generic_xgb(),
        grammarviz3(),
        # hybrid_knn(),  # no parameters to optimize
        img_embedding_cae(),
        # kmeans(),  # no parameters to optimize
        # laser_dbn(),  # no parameters to optimize
        # left_stampi(),  # no parameters to optimize
        lstm_ad(),
        median_method(),
        multi_hmm(),
        # norma(),  # no parameters to optimize
        normalizing_flows(),
        numenta_htm(),
        ocean_wnn(),
        pci(),
        phasespace_svm(),
        pst(),
        random_black_forest(),
        # robust_pca(),  # no parameters to optimize
        sand(),
        series2graph(),
        sr(),
        # stamp(),  # no parameters to optimize
        # stomp(),  # no parameters to optimize
        subsequence_if(),
        subsequence_lof(),
        tarzan(),
        telemanom(),
        ts_bitmap(),
    ]
    print(f"Selecting algorithms, page {algo_page + 1} of {len(algorithms) // algo_page_size + 1}:")
    algorithms = algorithms[algo_page * algo_page_size:(algo_page + 1) * algo_page_size]
    print(", ".join(a.name for a in algorithms))

    print("Configuring algorithms...")
    configurator.configure(algorithms,
                           ignore_shared=False,  # use already optimized shared parameters
                           perform_search=True,  # but perform search over the optimized parameter search spaces
                           assume_parameter_independence=True
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
