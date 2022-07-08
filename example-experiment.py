#!/usr/bin/env python3
import logging
import random
import sys
from pathlib import Path

from durations import Duration

from timeeval import TimeEval, DatasetManager, RemoteConfiguration, ResourceConstraints, DefaultMetrics
from timeeval.constants import HPI_CLUSTER
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
from timeeval_experiments.algorithms import *


# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.INFO,
    force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)


random.seed(42)


def main():
    # dm = DatasetManager(HPI_CLUSTER.akita_benchmark_path, create_if_missing=False)
    dm = DatasetManager(Path("tests/example_data"), create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="timeeval_experiments/param-config.example.json")

    # Select datasets and algorithms
    datasets = dm.select()
    datasets = random.sample(datasets, 1)
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        # generic_rf(),
        norma(),
        # sr_cnn(),
        # knn(),
        # cblof(),
        # hif(),
        # fft(),
        # dbstream(),
        # img_embedding_cae(),
        # telemanom(),
        # dwt_mlead(),
        # stamp(),
        # grammarviz3(),
        # hybrid_knn(),
        # normalizing_flows(),
        # ts_bitmap(),
        # lof(),
        # deepnap(),
        # multi_hmm(),
        # iforest(),
        # valmod(),
        # sarima(),
        # dspot(),
        # ssa(),
        # kmeans(),
        # hbos(),
        # encdec_ad(),
        # numenta_htm(),
        # pcc(),
        # novelty_svr(),
        # ensemble_gi(),
        # series2graph(),
        # mscred(),
        # copod(),
        # median_method(),
        # arima(),
        # torsk(),
        # tarzan(),
        # if_lof(),
        # cof(),
        # random_black_forest(),
        # fast_mcd(),
        # phasespace_svm(),
        # eif(),
        # tanogan(),
        stomp(),
        # hotsax(),
        # pci(),
        # robust_pca(),
        # dae(),
        # ocean_wnn(),
        # health_esn(),
        # lstm_ad(),
        # laser_dbn(),
        # deepant(),
        # bagel(),
        # generic_xgb(),
        # mtad_gat(),
        # omnianomaly(),
        # pst(),
        # donut(),
        # sr(),
        # autoencoder(),
    ]
    print(f"Selected algorithms: {len(algorithms)}")
    sys.stdout.flush()

    configurator.configure(algorithms, ignore_dependent=False, perform_search=False)
    for algo in algorithms:
        print(f"Algorithm {algo.name} param_grid:")
        for config in algo.param_config:
            print(f"  {config}")

    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=HPI_CLUSTER.nodes
    )
    limits = ResourceConstraints(
        tasks_per_host=15,
        task_cpu_limit=1.,
        train_timeout=Duration("1 minute"),
        execute_timeout=Duration("1 minute")
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        # distributed=True,
                        # remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.RANGE_PR_AUC]
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=False))


if __name__ == "__main__":
    main()
