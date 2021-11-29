#!/usr/bin/env python
import numpy as np
import pandas as pd

import tqdm

from timeeval import Datasets, DatasetRecord
from timeeval.constants import HPI_CLUSTER


def main():
  old_dmgr = Datasets(HPI_CLUSTER.akita_benchmark_path)
  dmgr = Datasets(".")

  existing_datasets = dmgr.select()
  datasets = [
    dataset
    for dataset in old_dmgr.select(collection="LTDB") + old_dmgr.select(collection="Kitsune") + old_dmgr.select(
          collection="GHL")
    if dataset not in existing_datasets
  ]
  print(f"Filling dataset index table with {len(datasets)} datasets")

  for dataset in tqdm.tqdm(datasets, desc="Processing"):
    tqdm.tqdm.write(f"Preparing dataset {dataset}")
    base_data = old_dmgr.df().loc[dataset]

    df = old_dmgr.get_dataset_df(dataset, train=False)
    l = len(df)
    dimensions = len(df.columns) - 2
    contamination = len(df[df["is_anomaly"] == 1]) / l

    means = df.iloc[:, 1:-1].mean(axis=0)
    stddevs = df.iloc[:, 1:-1].std(axis=0)
    mean = np.mean(means.values)
    stddev = np.mean(stddevs.values)

    labels = df["is_anomaly"]
    label_groups = labels.groupby((labels.shift() != labels).cumsum())
    anomalies = [len(v) for k, v in label_groups if np.all(v)]
    min_anomaly_length = np.min(anomalies) if anomalies else 0
    median_anomaly_length = int(np.median(anomalies)) if anomalies else 0
    max_anomaly_length = np.max(anomalies) if anomalies else 0
    num_anomalies = len(anomalies)

    dmgr.add_dataset(DatasetRecord(
      collection_name=dataset[0],
      dataset_name=dataset[1],
      train_path=base_data.train_path,
      test_path=base_data.test_path,
      dataset_type=base_data.dataset_type,
      datetime_index=base_data.datetime_index,
      split_at=base_data.split_at,
      train_type=base_data.train_type,
      train_is_normal=base_data.train_is_normal,
      input_type=base_data.input_type,

      length=l,
      dimensions=dimensions,
      contamination=contamination,
      num_anomalies=num_anomalies,
      min_anomaly_length=min_anomaly_length,
      median_anomaly_length=median_anomaly_length,
      max_anomaly_length=max_anomaly_length,
      mean=mean,
      stddev=stddev,
      trend=None,
      stationarity=None,
    ))
    dmgr.save()

  print("Finished!")


if __name__ == "__main__":
  main()
