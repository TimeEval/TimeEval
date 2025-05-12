#!/usr/bin/env python3
import logging
import random
import sys
from pathlib import Path

from durations import Duration

from timeeval import TimeEval, DatasetManager, RemoteConfiguration, ResourceConstraints
from timeeval.algorithms import *
from timeeval.metrics import RocAUC, RangeRocAUC, RangePrVUS
from timeeval_experiments.algorithm_configurator import AlgorithmConfigurator

# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)


random.seed(42)


def main():
    dm = DatasetManager(Path("tests/example_data"), create_if_missing=False)
    configurator = AlgorithmConfigurator(
        config_path="timeeval_experiments/param-config.example.json"
    )

    # Select datasets and algorithms
    datasets = dm.select()
    datasets = random.sample(datasets, 1)
    print(f"\nSelected datasets: {len(datasets)}")

    algorithms = [
        # arima(),
        # autoencoder(),
        # bagel(),
        # cblof(),
        # cof(),
        # copod(),
        # dae(),
        # damp(),
        # dbstream(),
        # deepant(),
        # deepnap(),
        # donut(),
        # dspot(),
        dwt_mlead(),
        # eif(),
        # encdec_ad(),
        # ensemble_gi(),
        # fast_mcd(),
        # fft(),
        # generic_rf(),
        # generic_xgb(),
        # grammarviz3(),
        # grammarviz3_multi(),
        # hbos(),
        # health_esn(),
        # hif(),
        # hotsax(),
        # hybrid_knn(),
        # if_lof(),
        # iforest(),
        # img_embedding_cae(),
        # kmeans(),
        # knn(),
        # laser_dbn(),
        # left_stampi(),
        lof(),
        # lstm_ad(),
        # lstm_vae(),
        # median_method(),
        # mscred(),
        # mstamp(),
        # mtad_gat(),
        # multi_hmm(),
        # multi_norma(),
        # multi_subsequence_lof(),
        # mvalmod(),
        # norma(),
        # normalizing_flows(),
        # novelty_svr(),
        # numenta_htm(),
        # ocean_wnn(),
        # omnianomaly(),
        # pcc(),
        # pci(),
        # phasespace_svm(),
        # pst(),
        # random_black_forest(),
        # robust_pca(),
        # s_h_esd(),
        # sand(),
        # sarima(),
        # series2graph(),
        # sr(),
        # sr_cnn(),
        # ssa(),
        # stamp(),
        stomp(),
        # subsequence_fast_mcd(),
        # subsequence_if(),
        # subsequence_knn(),
        # subsequence_lof(),
        # tanogan(),
        # tarzan(),
        # telemanom(),
        # torsk(),
        # triple_es(),
        # ts_bitmap(),
        # valmod(),
    ]
    print(f"Selected algorithms: {len(algorithms)}")
    configurator.configure(algorithms, ignore_dependent=False, perform_search=False)

    print()
    for algo in algorithms:
        print(f"Algorithm {algo.name} param_grid:")
        for config in algo.param_config.iter(algo, dataset=datasets[0]):
            print(f"  {config}")
    sys.stdout.flush()

    cluster_config = RemoteConfiguration(
        scheduler_host="localhost", worker_hosts=["localhost"]
    )
    limits = ResourceConstraints(
        tasks_per_host=1,
        task_cpu_limit=1.0,
        train_timeout=Duration("1 minute"),
        execute_timeout=Duration("1 minute"),
    )
    timeeval = TimeEval(
        dm,
        datasets,
        algorithms,
        distributed=True,
        remote_config=cluster_config,
        resource_constraints=limits,
        metrics=[
            RocAUC(),
            RangeRocAUC(buffer_size=100),
            RangePrVUS(max_buffer_size=100),
        ],
    )
    timeeval.run()
    print(timeeval.get_results(aggregated=False))


if __name__ == "__main__":
    main()
