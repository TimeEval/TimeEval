# Time series datasets

TimeEval uses a canonical file format for datasets.
Existing datasets in another format must first be transformed into the canonical format before they can be used with TimeEval.

## Canonical file format

TimeEval's canonical file format is based on CSV.
Each file requires a header, cells (values) are separated by commas (decimal seperator is `.`), and records are separated by newlines (unix-style LF: `\n`).
The first column of the dataset is its index, either in integer- or datetime-format
(multiple timestamp-formats are supported but [RFC 3339](https://tools.ietf.org/html/rfc3339) is preferred, e.g. `2017-03-22 15:16:45.433502912`).
The index follows a single or multiple (if multivariate dataset) time series columns.
The last column contains the annotations, `0` for normal points and `1` for anomalies.
Usage of the `timestamp` and `is_anomaly` column headers is recommended.

```
timestamp,value,is_anomaly
0,12751.0,1
1,8767.0,0
2,7005.0,0
3,5257.0,0
4,4189.0,0
```

## Registering datasets

TimeEval comes with its own collection of benchmark datasets (**currently not included**, download them [from our website](https://hpi-information-systems.github.io/timeeval-evaluation-paper/notebooks/Datasets.html)).
They can directly be used using the dataset manager `DatasetManager`:

```python
from pathlib import Path

from timeeval import DatasetManager
from timeeval.constants import HPI_CLUSTER

datasets_folder: Path = HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK]  # or Path("./datasets-folder")
dm = DatasetManager(datasets_folder)
datasets = dm.select()
```

### Custom datasets

```{important}
WIP!
```

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

You can register custom datasets at the dataset manager using two ways:

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

## Preparing datasets for TimeEval

Datasets in different formats should be transformed in TimeEval's canonical file format.
TimeEval provides a utility script to perform this transformation: `scripts/preprocess_dataset.py`.
You can download this scrip from its [GitHub repository](https://github.com/HPI-Information-Systems/TimeEval).

A single dataset can be provided in two Numpy-readable text files.
The first text file contains the data.
The labels must be in a separate text file.
Hereby, the label file can either contain the actual labels for each point in the data file or only the line indices of the anomalies.
Example source data files:

Data file

```
12751.0
8767.0
7005.0
5257.0
4189.0
```

Labels file (actual labels)

```
1
0
0
0
0
```

Labels file (line indices)

```
3
4
```

The script `scripts/preprocess_dataset.py` automatically generates the index column using an auto-incrementing integer value.
The integer value can be substituted with a corresponding timestamp (auto-incrementing value is used as a time unit, such as seconds `s` or hours `h` from the unix epoch).
See the tool documentation for further information:

```bash
python scripts/preprocess_dataset.py --help
```
