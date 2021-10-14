#!/usr/bin/env python3
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from durations import Duration
from sklearn.preprocessing import MinMaxScaler
from timeeval.params import IndependentParameterGrid

from timeeval import TimeEval, Datasets, Algorithm, TrainingType, InputDimensionality, AlgorithmParameter
from timeeval.adapters import FunctionAdapter
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
    force=True,
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
)

random.seed(42)
np.random.rand(42)


def main():
    dm = Datasets("/home/phillip/Datasets/GutenTAG/test-cases")
    configurator = AlgorithmConfigurator(config_path="param-config.json")

    # Select datasets and algorithms
    #datasets = dm.select()
    #datasets = [(collection, name) for (collection, name) in datasets if not name.startswith("cbf-")]
    datasets = [("GutenTAG", "sinus-diff-count-5.semi-supervised")]
    # datasets = random.sample(datasets, 200)
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        arima(),
        # autoencoder(),
        # bagel(),
        # cblof(),
        # cof(),
        # copod(),
        # dae(),
        # dbstream(),
        # deepant(),
        deepnap(),
        # donut(),
        # dspot(),
        # dwt_mlead(),
        # eif(),
        # encdec_ad(skip_pull=True),
        # ensemble_gi(),
        # fast_mcd(),
        # fft(),
        # generic_rf(),
        # generic_xgb(),
        # grammarviz3(),
        # hbos(),
        # health_esn(),
        # hif(),
        # hotsax(),
        # hybrid_knn(),
        # iforest(),
        # if_lof(),
        # img_embedding_cae(),
        # kmeans(),
        # knn(),
        # laser_dbn(),
        # left_stampi(),
        # lof(),
        # lstm_ad(),
        # lstm_vae(),
        # median_method(),
        # mscred(),
        # mtad_gat(),
        # multi_hmm(),
        # norma(),
        # normalizing_flows(),
        # novelty_svr(),
        numenta_htm(params=IndependentParameterGrid({
            "globalDecay": [0, 0.1, 0.5],
            "encoding_output_width": [25, 50, 75],
            "encoding_input_width": [15, 21, 30],
            "columnCount": [1024, 2048, 4096],
            "cellsPerColumn": [16, 32, 64],
            "autoDetectWaitRecords": [25, 50, 75],
            "activationThreshold": [6, 12, 24],
            "inputWidth": [1024, 2048, 4096],
            "initialPerm": [0.15, 0.21, 0.3],
            "maxAge": [0, 5, 10],
            "synPermConnected": [0.05, 0.1, 0.2],
            "synPermInactiveDec": [0.001, 0.005, 0.01],
            "synPermActiveInc": [0.05, 0.1, 0.2],
            "maxSegmentsPerCell": [64, 128, 256],
            "potentialPct": [0.1, 0.5, 0.9],
            "permanenceInc": [0.05, 0.1, 0.2],
            "permanenceDec": [0.05, 0.1, 0.2],
            "pamLength": [1, 3, 5],
            "numActiveColumnsPerInhArea": [30, 40, 50],
            "newSynapseCount": [15, 20, 30],
            "minThreshold": [6, 9, 12],
            "maxSynapsesPerSegment": [16, 32, 64]
        })),
        # ocean_wnn(),
        # omnianomaly(),
        # pcc(),
        # pci(),
        # phasespace_svm(),
        # pst(),
        # random_black_forest(),
        # robust_pca(),
        # sarima(),
        # series2graph(),
        # sr(),
        # sr_cnn(),
        # ssa(),
        # stamp(),
        # stomp(),
        # subsequence_fast_mcd(),
        # subsequence_if(),
        # subsequence_lof(),
        # s_h_esd(),
        # tanogan(),
        # tarzan(),
        # telemanom(),
        # torsk(),
        # triple_es(),
        # ts_bitmap(),
        # valmod()
    ]

    print(f"Selected algorithms: {len(algorithms)}\n\n")
    sys.stdout.flush()

    configurator.configure(algorithms, perform_search=True, assume_parameter_independence=True)

    cluster_config = RemoteConfiguration(
        scheduler_host="localhost",
        worker_hosts=["localhost", "localhost", "localhost", "localhost"]
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
        task_cpu_limit=1.,
        train_fails_on_timeout=False,
        train_timeout=Duration("30 minutes"),
        execute_timeout=Duration("10 minutes"),
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=False,
                        remote_config=cluster_config,
                        resource_constraints=limits,
                        force_training_type_match=True,
                        metrics=[Metric.ROC_AUC, Metric.PR_AUC, Metric.RANGE_PR_AUC, Metric.AVERAGE_PRECISION],
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True, short=True))


if __name__ == "__main__":
    main()
