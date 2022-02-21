#!/usr/bin/env python3
import logging
import random
import sys

from durations import Duration

from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints
from timeeval.utils.metrics import Metric
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


def main():
    dm = Datasets(HPI_CLUSTER.akita_test_case_path, create_if_missing=False)
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    datasets = dm.select()
    # datasets = random.sample(datasets, 200)
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        arima(),
        autoencoder(),
        bagel(),
        cblof(),
        cof(),
        copod(),
        dae(),
        dbstream(),
        deepant(),
        deepnap(),
        donut(),
        dspot(),
        dwt_mlead(),
        eif(),
        encdec_ad(),
        ensemble_gi(),
        fast_mcd(),
        fft(),
        generic_rf(),
        generic_xgb(),
        grammarviz3(),
        hbos(),
        health_esn(),
        hif(),
        hotsax(),
        hybrid_knn(),
        iforest(),
        if_lof(),
        img_embedding_cae(),
        kmeans(),
        knn(),
        laser_dbn(),
        left_stampi(),
        lof(),
        lstm_ad(),
        median_method(),
        mscred(),
        mtad_gat(),
        multi_hmm(),
        norma(),
        normalizing_flows(),
        novelty_svr(),
        numenta_htm(),
        ocean_wnn(),
        omnianomaly(),
        pcc(),
        pci(),
        phasespace_svm(),
        pst(),
        random_black_forest(),
        robust_pca(),
        sarima(),
        series2graph(),
        sr(),
        sr_cnn(),
        ssa(),
        stamp(),
        stomp(),
        subsequence_if(),
        subsequence_lof(),
        s_h_esd(),
        tanogan(),
        tarzan(),
        telemanom(),
        torsk(),
        triple_es(),
        ts_bitmap(),
        valmod(),
    ]
    print(f"Selected algorithms: {len(algorithms)}\n\n")
    sys.stdout.flush()

    configurator.configure(algorithms, perform_search=False)

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
                        metrics=[Metric.ROC_AUC, Metric.RANGE_PR_AUC],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
