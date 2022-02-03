#!/usr/bin/env python3
import logging
from pathlib import Path

from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints
from timeeval_experiments.algorithms import *

# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.INFO,
    # force=True,
    # encoding="UTF-8",
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
    # datefmt="%y-%m%d %H:M:%S",
)


def main():
    rr_data = Path("/home/projects/akita/data/01_20201211_UseCase_FZG/03_Analysis/stgg_anomaly/datasets.json")
    dm = Datasets(HPI_CLUSTER.akita_benchmark_path, custom_datasets_file=rr_data)

    # Select datasets and algorithms
    benchmark_datasets = dm.select(train_type="unsupervised", input_type="univariate")
    rr_datasets = dm.select(collection_name="custom")
    datasets = benchmark_datasets + rr_datasets
    print(f"Selected datasets: {len(datasets)}")

    algorithms = [
        arima(),
        cblof(),
        cof({"n_neighbors": [10, 20]}),
        copod(),
        eif({"n_trees": [100, 500]}),
        fft({"fft_parameters": [2, 5, 10]}),
        grammarviz(),
        hbos(),
        hotsax(),
        iforest({"n_estimators": [100, 500]}),
        knn(),
        lof({"n_neighbors": [10, 20]}),
        norma({"dummy": range(3)}),  # force repetitions just for norma using a dummy parameter (to find good rand seed)
        numenta_htm(),
        pcc(),
        pci(),
        phasespace_svm({"project_phasespace": [True, False]}),
        knn({"n_neighbors": [10, 20]}),
        series2graph({"window_size": [30, 50]}),
        stamp({"window_size": [30, 50]}),
        stomp({"window_size": [30, 50]}),
        # torsk(), # has some errors
        ts_bitmap(),
        valmod(),
        valmod([
            {"window_min": [30], "window_max": [50]},
            {"window_min": [50], "window_max": [70]},
        ]),
    ]
    print(f"Selected algorithms: {list(map(lambda algo: algo.name, algorithms))}")

    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=HPI_CLUSTER.nodes
    )
    limits = ResourceConstraints(
        tasks_per_host=15,
        task_cpu_limit=1.,
    )
    timeeval = TimeEval(dm, datasets, algorithms,
                        repetitions=1,
                        distributed=True,
                        remote_config=cluster_config,
                        resource_constraints=limits
                        )

    timeeval.run()
    print(timeeval.get_results(aggregated=True))


if __name__ == "__main__":
    main()
