#!/usr/bin/env python3
import logging
import random
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import MultiDatasetManager, RemoteConfiguration, TimeEval
from timeeval.algorithms import *
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import DefaultMetrics, RangePrAUC, RangeRocAUC
from timeeval.resource_constraints import GB, ResourceConstraints
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator
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
MAX_CONTAMINATION = 0.1
MIN_ANOMALIES = 1


def main():
    dm = MultiDatasetManager([
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK],
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.UNIVARIATE_ANOMALY_TEST_CASES],
    ])
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
    datasets += dm.select(collection="TODS-synthetic",
                          max_contamination=MAX_CONTAMINATION,
                          min_anomalies=MIN_ANOMALIES)

    print(f"Selecting {len(datasets)} datasets")

    algorithms = [
        arima(),
        # autoencoder(),  # exclude
        bagel(),
        cblof(),
        cof(),
        copod(),
        # dae(),  # exclude
        dbstream(),
        deepant(),
        # deepnap(),  # run later with less datasets
        donut(),
        dspot(),
        dwt_mlead(),
        eif(),
        encdec_ad(),
        # ensemble_gi(),  # exclude
        # fast_mcd(),  # exclude
        fft(),
        generic_rf(),
        generic_xgb(),
        grammarviz3(),
        hbos(),
        health_esn(),
        hif(),
        hotsax(),
        hybrid_knn(),
        if_lof(),
        iforest(),
        img_embedding_cae(),
        kmeans(),
        knn(),
        laser_dbn(),
        left_stampi(),
        lof(),
        lstm_ad(),
        # lstm_vae(),  # exclude
        median_method(),
        # mscred(),  # exclude
        # mtad_gat(),  # exclude
        multi_hmm(),
        norma(),
        normalizing_flows(),
        # novelty_svr(),  # exclude
        numenta_htm(),
        ocean_wnn(),
        omnianomaly(),
        pcc(),
        pci(),
        phasespace_svm(),
        pst(),
        random_black_forest(),
        robust_pca(),
        s_h_esd(),
        sand(),
        # sarima(),  # exclude
        series2graph(),
        sr(),
        sr_cnn(),
        ssa(),
        stamp(),
        stomp(),
        # subsequence_fast_mcd(),  # exclude
        subsequence_if(),
        subsequence_lof(),
        tanogan(),
        tarzan(),
        telemanom(),
        torsk(),
        triple_es(),
        ts_bitmap(),
        valmod(),
        Baselines.normal()
    ]
    print(f"Selecting {len(algorithms)} algorithms")

    print("Configuring algorithms...")
    configurator.configure(algorithms, perform_search=False)

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
    sys.stdout.flush()


    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=list(set(HPI_CLUSTER.nodes) - {HPI_CLUSTER.odin14, HPI_CLUSTER.odin13, HPI_CLUSTER.odin12}),
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=6 * GB,
        train_timeout=Duration("4 hours"),
        execute_timeout=Duration("4 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[
                            DefaultMetrics.ROC_AUC, DefaultMetrics.PR_AUC,
                            RangeRocAUC(buffer_size=100), RangePrAUC(buffer_size=100)
                        ])

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
