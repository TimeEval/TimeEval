#!/usr/bin/env python3
import logging
import random
import shutil
import sys
from typing import List, Tuple

import numpy as np
from durations import Duration

from timeeval import RemoteConfiguration
from timeeval import TimeEval, DatasetManager
from timeeval.constants import HPI_CLUSTER
from timeeval.metrics import DefaultMetrics, RangePrVUS, RangeRocVUS
from timeeval.params import FixedParameters
from timeeval.resource_constraints import ResourceConstraints, GB
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
MAX_CONTAMINATION = 0.1
MIN_ANOMALIES = 1


def main():
    dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK], create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = []
    datasets += dm.select(collection="TSB-UAD-synthetic", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
    datasets += dm.select(collection="TSB-UAD-artificial", max_contamination=MAX_CONTAMINATION, min_anomalies=MIN_ANOMALIES)
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
        # Baselines.normal(),  # exclude because no additional value!
        # additional algorithms:
        subsequence_knn(),
    ]

    print("Configuring algorithms...")
    configurator.configure(algorithms, perform_search=False)

    # preconfigured additional algorithms:
    algorithms.append(multi_subsequence_lof(FixedParameters({
        "dim_aggregation_method": "concat",
        "window_size": "heuristic:PeriodSizeHeuristic(factor=1.0, fb_value=100)",
        "distance_metric_order": 2,
        "leaf_size": 20,
        "n_jobs": 1,
        "n_neighbors": 50,
        "random_state": 42
    })))

    print(f"Selecting {len(algorithms)} algorithms")

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
        worker_hosts=HPI_CLUSTER.nodes
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        task_memory_limit=3*GB,
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
                        force_dimensionality_match=False,
                        force_training_type_match=False,
                        metrics=[DefaultMetrics.ROC_AUC, DefaultMetrics.PR_AUC, DefaultMetrics.FIXED_RANGE_PR_AUC,
                                 RangePrVUS(), RangeRocVUS()],
                        )

    # copy parameter configuration file to results folder
    timeeval.results_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(configurator.config_path, timeeval.results_path)

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
