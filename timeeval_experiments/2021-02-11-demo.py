#!/usr/bin/env python3
from timeeval_experiments.algorithms import *
from timeeval import TimeEval, Datasets
from timeeval.constants import HPI_CLUSTER


def main():
    dm = Datasets(HPI_CLUSTER.akita_benchmark_path)
    # dm = Datasets("../tests/example_data")

    # Select datasets and algorithms
    datasets = dm.select(collection_name="NAB", train_type="unsupervised", input_type="univariate")
    # datasets = dm.select(collection_name="test", train_type="unsupervised", input_type="univariate")
    print(f"Selected datasets: {datasets}")

    algorithms = [
        stamp({"window_size": [30, 40]}),
        stomp({"window_size": [30, 40]}),
        valmod([
            {"verbose": [2], "window_min": [30]},
            {"verbose": [1], "window_min": [40]},
        ]),
        lof(),
        cof({"n_neighbors": [5]}),
        series2graph({"convolution_size": [8, 16]}),
        norma(),
        grammarviz(),
        hotsax({"window_size": [100, 120]}),
    ]
    print(f"Selected algorithms: {list(map(lambda algo: algo.name, algorithms))}")

    cluster_config = {
        "hosts": HPI_CLUSTER.nodes,
        "remote_python": "/home/sebastian.schmidl/.conda/envs/timeeval/bin/python"
    }
    timeeval = TimeEval(dm, datasets, algorithms, repetitions=1, distributed=True, ssh_cluster_kwargs=cluster_config)

    timeeval.run()
    print(timeeval.get_results(aggregated=False))


if __name__ == "__main__":
    main()
