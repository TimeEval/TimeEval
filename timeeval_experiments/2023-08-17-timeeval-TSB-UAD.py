#!/usr/bin/env python3
import logging
import shutil
import sys
import random
from typing import List, Tuple

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
        HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK],
    ])
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets: List[Tuple[str, str]] = [
        ("TSB-UAD-synthetic", "YAHOO_flat_region_0.02-Yahoo_A2synthetic_41_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_resampling_0.04-Yahoo_A2synthetic_32_data"),
        ("TSB-UAD-synthetic",
         "KDD21_flip_segment_0.02-210_UCR_Anomaly_Italianpowerdemand_36123_74900_74996"),
        ("TSB-UAD-synthetic", "YAHOO_flip_segment_0.08-YahooA4Benchmark-TS55_data"),
        ("TSB-UAD-synthetic", "YAHOO_flat_region_0.08-YahooA4Benchmark-TS33_data"),
        ("TSB-UAD-synthetic", "YAHOO_flat_region_0.02-Yahoo_A1real_14_data"),
        ("TSB-UAD-synthetic",
         "KDD21_flat_region_0.02-104_UCR_Anomaly_NOISEapneaecg4_6000_16000_16100"),
        ("TSB-UAD-synthetic",
         "KDD21_change_segment_resampling_0.02-033_UCR_Anomaly_DISTORTEDInternalBleeding5_4000_6200_6370"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_resampling_0.02-YahooA3Benchmark-TS84_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_add_scale_0.02-YahooA3Benchmark-TS22_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_add_random_walk_trend_0.2-YahooA3Benchmark-TS75_data"),
        ("TSB-UAD-synthetic",
         "KDD21_change_segment_resampling_0.02-006_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature2_4000_5703_5727"),
        ("TSB-UAD-synthetic", "YAHOO_add_random_walk_trend_0.2-Yahoo_A1real_32_data"),
        ("TSB-UAD-synthetic", "YAHOO_flat_region_0.08-Yahoo_A2synthetic_1_data"),
        ("TSB-UAD-synthetic",
         "KDD21_change_segment_resampling_0.02-074_UCR_Anomaly_DISTORTEDqtdbSel1005V_4000_12400_12800"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_resampling_0.08-Yahoo_A2synthetic_93_data"),
        ("TSB-UAD-synthetic", "YAHOO_flat_region_0.04-YahooA3Benchmark-TS51_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_add_random_walk_trend_0.2-YahooA4Benchmark-TS28_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_resampling_0.04-YahooA3Benchmark-TS85_data"),
        ("TSB-UAD-synthetic",
         "YAHOO_change_segment_add_scale_0.02-YahooA4Benchmark-TS28_data"),
        ("TSB-UAD-artificial", "-61_2_0.02_25"),
        ("TSB-UAD-artificial", "-69_2_0.02_25"),
        ("TSB-UAD-artificial", "-107_2_0.02_11"),
        ("TSB-UAD-artificial", "-86_2_0.02_35"),
        ("TSB-UAD-artificial", "-37_2_0.02_25"),
        ("TSB-UAD-artificial", "-72_2_0.01_5"),
        ("TSB-UAD-artificial", "-63_2_0.02_15"),
        ("TSB-UAD-artificial", "-116_2_0.02_9"),
        ("TSB-UAD-artificial", "-34_2_0.02_15"),
        ("TSB-UAD-artificial", "-11_2_0.02_6"),
        ("TSB-UAD-artificial", "-55_2_0.02_6"),
        ("TSB-UAD-artificial", "-116_2_0.01_5"),
        ("TSB-UAD-artificial", "-6_2_0.02_15"),
        ("TSB-UAD-artificial", "-24_2_0.02_3"),
        ("TSB-UAD-artificial", "-104_2_0.02_35"),
        ("TSB-UAD-artificial", "-40_2_0.02_25"),
        ("TSB-UAD-artificial", "-94_2_0.01_5"),
        ("TSB-UAD-artificial", "-69_2_0.02_15"),
        ("TSB-UAD-artificial", "-48_2_0.02_11"),
        ("TSB-UAD-artificial", "-25_2_0.02_9"),
    ]

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
        task_memory_limit=3 * GB,
        use_preliminary_model_on_train_timeout=True,
        use_preliminary_scores_on_execute_timeout=True,
        train_timeout=Duration("2 hours"),
        execute_timeout=Duration("2 hours"),
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
