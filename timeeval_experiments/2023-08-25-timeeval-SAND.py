#!/usr/bin/env python3
import logging
import shutil
import sys
import random
from typing import List, Tuple
from pathlib import Path

import numpy as np
from durations import Duration

from timeeval import TimeEval, MultiDatasetManager, RemoteConfiguration, ResourceConstraints
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import RangePrAUC, RangeRocAUC, DefaultMetrics
from timeeval.resource_constraints import GB
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
    dm = MultiDatasetManager([
        Path("/home/projects/akita/data/sand-data"),
        # Path("../../data/sand-data/processed/timeeval"),
    ])
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Minimum list of datasets:
    # datasets: List[Tuple[str, str]] = [
    #     ("SAND", "803"),
    #     ("SAND", "806"),
    #     ("SAND", "803_820"),
    #     ("SAND", "803_806"),
    #     ("SAND", "805_806_820"),
    #     ("SAND", "803_805_820"),
    #     ("SAND", "803_805_806_820"),
    #     ("SAND", "SED"),
    #     ("SAND", "803_SED"),
    #     ("SAND", "806_SED"),
    #     ("SAND", "803_806_820_SED"),
    #     ("SAND", "806_SED"),
    #     ("SAND", "803_SED"),
    #     ("SAND", "SED"),
    #     ("SAND", "SED"),
    #     ("SAND", "803_806_820_SED"),
    #     ("SAND", "806_SED"),
    # ]

    # Select datasets and algorithms
    datasets = dm.select(collection="SAND")

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
    ]

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
        worker_hosts=HPI_CLUSTER.nodes,
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=6 * GB,
        use_preliminary_model_on_train_timeout=True,
        use_preliminary_scores_on_execute_timeout=True,
        train_timeout=Duration("4 hours"),
        execute_timeout=Duration("4 hours"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        skip_invalid_combinations=True,
                        metrics=[RangeRocAUC(buffer_size=100), RangePrAUC(buffer_size=100), DefaultMetrics.ROC_AUC, DefaultMetrics.PR_AUC],
                        )

    # copy parameter configuration file to results folder
    timeeval.results_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configurator.config_path, timeeval.results_path)

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
