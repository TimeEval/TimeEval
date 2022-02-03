#!/usr/bin/env python3
import logging
# from pathlib import Path

from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER
from timeeval.remote import RemoteConfiguration
from timeeval.resource_constraints import ResourceConstraints
from timeeval_experiments.algorithms import *

# Setup logging
logging.basicConfig(
    filename="timeeval.log",
    filemode="a",
    level=logging.DEBUG,
    # force=True,
    # encoding="UTF-8",
    format="%(asctime)s %(levelname)6.6s - %(name)20.20s: %(message)s",
    # datefmt="%y-%m%d %H:M:%S",
)


def main():
    dm = Datasets(HPI_CLUSTER.akita_benchmark_path)

    # Select datasets and algorithms
    datasets = dm.select(train_type="unsupervised", input_type="univariate")
    print(f"Selected datasets: {datasets}")

    algorithms = [
        arima(),
        copod(),
        norma({"dummy": range(5)}),  # force repetitions just for norma using a dummy parameter
        grammarviz3(),
        fft({"fft_parameters": [2, 5, 10]}),
        eif({"ntrees": [100, 200]}),
        knn({"n_neighbors": [5, 10]}),
        numenta_htm(),
        pcc({"whiten": [True, False]}),
        pci(),
        phasespace_svm({"project_phasespace": [True, False]})
    ]
    print(f"Selected algorithms: {list(map(lambda algo: algo.name, algorithms))}")

    cluster_config = RemoteConfiguration(
        scheduler_host=HPI_CLUSTER.odin01,
        worker_hosts=HPI_CLUSTER.nodes
    )
    limits = ResourceConstraints(
        tasks_per_host=10,
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
    # TimeEval.rsync_results(Path("/home/sebastian.schmidl/projects/timeeval/timeeval_experiments/results/2021_03_02_12_32_20"), HPI_CLUSTER.nodes)
