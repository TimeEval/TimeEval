# How to use your own datasets in TimeEval

You can use your own datasets with TimeEval.
There are two ways to achieve this: using **custom datasets** or preparing your datasets as a TimeEval dataset collection.
Either way, please familiarize yourself with the dataset format used by TimeEval described in the concept page [](../concepts/datasets.md).

## 1. Custom datasets

```{important}
The time series CSV-files must follow the [TimeEval canonical file format](../concepts/datasets.md#canonical-file-format)!
```

To tell the TimeEval tool where it can find your custom datasets, a configuration file is needed.
The custom datasets config file contains all custom datasets organized by their identifier which is used later on.
Each entry in the config file must contain the path to the test time series;
optionally, one can add a path to the training time series, specify the dataset type, and supply the period size if known.
The paths to the data files must be absolute or relative to the configuration file.
Example file `custom_datasets.json`:

```json
{
  "dataset_name": {
    "test_path": "/absolute/path/to/data.csv"
  },
  "other_supervised_dataset": {
    "test_path": "/absolute/path/to/test.csv",
    "train_path": "./train.csv",
    "type": "synthetic",
    "period": 20
  }
}
```

You can add custom datasets to the dataset manager using two ways:

```python
from pathlib import Path

from timeeval import DatasetManager
from timeeval.constants import HPI_CLUSTER

custom_datasets_path = Path("/absolute/path/to/custom_datasets.json")

# Directly during initialization
dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK], custom_datasets_file=custom_datasets_path)

# Later on
dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK])
dm.load_custom_datasets(custom_datasets_path)
```

## 2. Create a TimeEval dataset collection

```{warning}
WIP
```
