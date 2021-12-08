#!/usr/bin/env python3
from pathlib import Path

import tqdm

from timeeval import Datasets, TrainingType
from timeeval.constants import HPI_CLUSTER
from timeeval.datasets import DatasetAnalyzer, DatasetRecord


"""
Goes through all existing datasets and re-generates the metadata.
Overall metadata is then stored in the datasets index file.
"""

new_index_file_path = Path(".").absolute()
ignore_datasets = ["01_Lev_fault_Temp_corr_seed_11_vars_23"]
ignore_collections = ["Kitsune", "LTDB"]  # "MITDB", "GHL"]


def main():
  print(f"Reading datasets from {HPI_CLUSTER.akita_benchmark_path}")
  print(f"Writing metadata to the original directory and the index file to {new_index_file_path}")
  dmgr = Datasets(HPI_CLUSTER.akita_benchmark_path)
  dmgr_new = Datasets(new_index_file_path)

  print("###############################")
  print("Analyzing unsupervised datasets")
  print("###############################")
  unsupervised(dmgr, dmgr_new)

  print("###############################")
  print("Analyzing (semi-)supervised datasets")
  print("###############################")
  supervised(dmgr, dmgr_new)


def unsupervised(dmgr, dmgr_new):
  existing_datasets = dmgr_new.select()

  unsupervised_datasets = [
    dataset
    for dataset in dmgr.select(training_type=TrainingType.UNSUPERVISED)
    if dataset[0] not in ignore_collections
    if dataset[1] not in ignore_datasets
    if dataset not in existing_datasets
  ]
  train = False
  try:
    for dataset in tqdm.tqdm(unsupervised_datasets, desc="Analyzing unsupervised datasets"):
      meta_data = analyze(dmgr, dataset, train=False)
      add_to_new(dmgr, dmgr_new, dataset, meta_data)
  finally:
    dmgr_new.save()


def supervised(dmgr, dmgr_new):
  existing_datasets = dmgr_new.select()

  supervised_datasets = [
    dataset
    for dataset in dmgr.select(training_type=TrainingType.SUPERVISED) + dmgr.select(training_type=TrainingType.SEMI_SUPERVISED)
    if dataset[0] not in ignore_collections
    if dataset[1] not in ignore_datasets
    if dataset not in existing_datasets
  ]
  train = False
  for dataset in tqdm.tqdm(supervised_datasets, desc="Analyzing (semi-)supervised datasets"):
    try:
      test_meta_data = analyze(dmgr, dataset, train=False)
      _ = analyze(dmgr, dataset, train=True)
      add_to_new(dmgr, dmgr_new, dataset, test_meta_data)
    finally:
      dmgr_new.save()


def analyze(dmgr, dataset, train: bool = False):
  tqdm.tqdm.write(f"Analyzing dataset {dataset} ({'training' if train else 'testing'})")
  da = DatasetAnalyzer(dataset, is_train=train, dmgr=dmgr)
  meta_file = dmgr.get_dataset_path(dataset, train=train).parent / f"{dataset[1]}.{Datasets.METADATA_FILENAME_SUFFIX}"
  da.save_to_json(meta_file, overwrite=not train)
  return da.metadata


def add_to_new(dmgr, dmgr_new, dataset, meta_data):
  base_data = dmgr.df().loc[dataset]
  dmgr_new.add_dataset(DatasetRecord(
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
      length=meta_data.length,
      dimensions=meta_data.dimensions,
      contamination=meta_data.contamination,
      num_anomalies=meta_data.num_anomalies,
      min_anomaly_length=meta_data.anomaly_length.min,
      median_anomaly_length=meta_data.anomaly_length.median,
      max_anomaly_length=meta_data.anomaly_length.max,
      mean=meta_data.mean,
      stddev=meta_data.stddev,
      trend=meta_data.trend,
      stationarity=meta_data.get_stationarity_name(),
  ))


if __name__ == "__main__":
  main()
