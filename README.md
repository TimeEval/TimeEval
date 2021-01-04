# TimeEval

Evaluation Tool for Anomaly Detection Algorithms on Time Series

## Usage

### Datasets

A single dataset should be provided in a/two (with labels) Numpy-readable text file. The labels must be in a separate text file. Hereby, the label file can either contain the actual labels for each point in the data file or only the line indices of the anomalies.

#### Example

Dataset file
```csv
12751.0
8767.0
7005.0
5257.0
4189.0
```

Labels file (actual labels)
```csv
1
0
0
0
0
```

or Labels file (line indices)
```csv
0
```

#### Registering Dataset

To tell the TimeEval tool where it can find which dataset, a configuration file is needed that contains all required datasets organized by their identifier which is used later on.

Config file
```json
{
  "dataset_name": {
    "data": "dataset.ts",
    "labels": "labels.txt"
  }
}
```

### Algorithms

Any algorithm that can be called with a numpy array as parameter and a numpy array as return value can be evaluated. However, so far only __unsupervised__ algorithms are supported.

#### Registering Algorithm

```python
from timeeval import TimeEval, Algorithm
from pathlib import Path
import numpy as np

def my_algorithm(data: np.ndarray) -> np.ndarray:
    return np.zeros_like(data)

datasets = ["webscope", "mba", "eeg"]
algorithms = [
    # Add algorithms to evaluate...
    Algorithm(
        name="MyAlgorithm",
        function=my_algorithm,
        data_as_file=False
    )
]

timeeval = TimeEval(datasets, algorithms, dataset_config=Path("dataset.json"))
timeeval.run()
print(timeeval.results)
```