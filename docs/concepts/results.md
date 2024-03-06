# TimeEval results

On configuring and executing TimeEval, TimeEval applies the algorithms with their configured hyperparameter values on all the datasets.
It measures the algorithms' runtimes and checks their effectiveness using evaluation measures ({doc}`metrics <api/timeeval.metrics>`).
The results are stored in a summary file called `results.csv` and a nested folder structure in the results-folder (`./results/<timestamp>` per default).
The output directory has the following structure:

```bash
results/<timestamp>/
├── results.csv
├── <algorithm_1>/<hyper_params_id>/
|   ├── <collection_name>/<dataset_name_1>/<repetition_number>/
│   |   ├── raw_anomaly_scores.ts
│   |   ├── anomaly_scores.ts
│   |   ├── docker-algorithm-scores.csv
│   |   ├── execution.log
│   |   ├── hyper_params.json
│   |   └── metrics.csv
|   └── <collection_name>/<dataset_name_2>/<repetition_number>/
│       ├── raw_anomaly_scores.ts
│       ├── anomaly_scores.ts
│       ├── docker-algorithm-scores.csv
│       ├── execution.log
|       ├── model.pkl
│       ├── hyper_params.json
│       └── metrics.csv
└── <algorithm_2>/<hyper_params_id>/
    └── <collection_name>/<dataset_name_1>/<repetition_number>/
        ├── raw_anomaly_scores.ts
        ├── anomaly_scores.ts
        ├── docker-algorithm-scores.csv
        ├── execution.log
        ├── hyper_params.json
        └── metrics.csv
```

We provide a description of each file below.

## Summary file (`result.csv`)

For a given dataset, different algorithms with varying hyperparameters yield distinct results.
The file `result.csv` provides an overview of the evaluation run and contains the following attributes:

| Column Name | Datatype | Description |
| --- | --- | --- |
| algorithm| str | name of the algorithm as defined in {class}`~timeeval.Algorithm_name` attribute |
| collection| str | name of the dataset collection. A collection contains similar datasets. |
| dataset| str | name of the dataset |
| algo_training_type | str | specifies, whether a dataset has a training time series with anomaly labels (supervised), with normal data only (semi-supervised), or no training time series at all (unsupervised)|
| algo_input_dimensionality | str | specifies if the dataset has multiple channels (multivariate) or not (univariate) |
| dataset_training_type | str | specifies, whether an algorithm requires training data with anomalies (supervised), without normal data only (semi-supervised), or does not require training data (unsupervised) |
| dataset_input_dimensionality | str |univariate or multivariate (see above) |
| train_preprocess_time| float64| runtime of the preprocessing step during training in seconds|
| train_main_time| float64 | runtime of the training in seconds (does not include pre-processing time)|
| execute_preprocess_time| float64 | runtime of the preprocessing step during execution in seconds|
| execute_main_time | float64 | runtime of the execution of the algorithm on the test time series in seconds (does not include pre- or post-processing times)|
| execute_postprocess_time|  float64 | runtime of the post-processing step during execution|
| status| str | specifies, whether the algorithm executed successfully ({obj}`~timeeval.Status.OK`), exceeded the time limit ({obj}`~timeeval.Status.TIMEOUT`), exceeded the memory limit ({obj}`~timeeval.Status.OOM`), or failed ({obj}`~timeeval.Status.ERROR`) |
| error_message| str | optional detailed error message|
| repetition| int | repetition number if a dataset-hyperparameter-dataset combination was executed multiple times|
| hyper_params| float64 | actual hyperparameter values for this execution|
| hyper_params_id| float64 | alphanumerical hash of the hyperparameter configuration|
| metric_1| float64 | value of the first performance metric|
| ...   |  ... | ...  |

## Directory (`<algorithm_1>/<hyper_params_id>/<collection_name>/<dataset_name_1>/<repetition_number>/`)

For every experiment in the configured evaluation run, TimeEval creates a new directory in the result folder.
It stores all the results and temporary files for this single combination of dataset, algorithm, algorithm hyperparameter values, and repetition number.
The directories are structured in nested folders named by first the algorithm name, followed by the ID of the hyperparameter settings, the dataset collection name, the dataset name, and finally the repetition number.
Each experiment directory contains at least the following files:

- `raw_anomaly_scores.ts`:
  The raw anomaly scores produced by the algorithm after the post-processing function was executed.
  The file contains no header and a single floating point value in each row for each time step of the input time series.
  The value range depends on the algorithm.
- `anomaly_scores.ts`:
  Normalized anomaly scores.
  The value range is from 0 (normal) to 1 (most anomalous).
- `execution.log`:
  Unstructured log-file of the experiment execution.
  Contains debugging information from the Adapter, the algorithm, and the metric calculation.
  If an algorithm fails, its error message usually appear in this log.
- `metrics.csv`:
  This file lists the metric and runtime measurements for the corresponding experiment.
  The used metrics are defined by the user.
  Find more information in the API documentation: {doc}`api/timeeval.metrics`
- `hyper_params.json`:
  Contains a JSON-object with the hyperparameter values used to execute the algorithm on the dataset.
  If hyperparameter heuristics were defined, the heuristic' values are already resolved.

All other files are optional and depend on the used {class}`~timeeval.adapters.BaseAdapter`.
For example, the {class}`~timeeval.adapters.DockerAdapter` usually produces a temporary file called `docker-algorithm-scores.csv`
to pass the algorithm result from the Docker container to TimeEval, and (semi-)supervised algorithms store their trained model in `model.pkl`-files.
