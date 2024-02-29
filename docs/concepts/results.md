# TimeEval results

On configuring and executing TimeEval, TimeEval applies the algorithms with their configured hyperparameter values on all the datasets.
It measures the algorithms' runtimes and checks their effectiveness using evaluation measures (:doc:`metrics <api/timeeval.metrics>`).
The results are stored in a summary file `results.csv`  and a nested folder structure in a results folder (``./results/<timestamp>`` per default).
The output directory has the following structure:

```bash
results/<time-stamp>/
├── results.csv
├── <algorithm_1>/<hyper_params_id>/
|   ├── <collection_name>/<dataset_name_1>/<repetition_number>/
│   |   ├── anomaly_scores.ts
│   |   ├── docker-algorithm-scores.csv
│   |   ├── execution.log
│   |   ├── hyper_params.json
│   |   └── metrics.csv
|   └── <collection_name>/<dataset_name_2>/<repetition_number>/
│       ├── anomaly_scores.ts
│       ├── docker-algorithm-scores.csv
│       ├── execution.log
|       ├── model.pkl
│       ├── hyper_params.json
│       └── metrics.csv
└── <algorithm_2>/<hyper_params_id>/
    └── <collection_name>/<dataset_name_1>/<repetition_number>/
        ├── anomaly_scores.ts
        ├── docker-algorithm-scores.csv
        ├── execution.log
        ├── hyper_params.json
        └── metrics.csv

```
Description of each file is given below. 

## result.csv

For a given dataset, different algorithms with varying hyperparameters yield distict results. The file `result.csv` provides an overview of evalution and contains following attributes:

| Column Name | Datatype | Description |
| --- | --- | --- |
| algorithm| str | name of the algorithm as defined in :class:`~timeeval.Algorithm_name` attribute |
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
| status| str | specifies, whether the algorithm executed successfully (:obj:`~timeeval.Status.OK`), exceeded the time limit (:obj:`~timeeval.Status.TIMEOUT`), exceeded the memory limit (:obj:`~timeeval.Status.OOM`), or failed (:obj:`~timeeval.Status.ERROR`) |
| error_message| str | optional detailed error message|
| repetition| int | repetition number if a dataset-hyperparameter-dataset combination was executed multiple times|
| hyper_params| float64 | actual hyperparameter values for this execution|
| hyper_params_id| float64 | alphanumerical hash of the hyperparameter configuration|
| metric_1| float64 | value of the first performance metric|
| ...   |  ... | ...  |

## Directory - <algorithm_1>/<hyper_params_id>/<collection_name>/<dataset_name_1>/<repetition_number>/

For every modification in hyperparameter of algorithms, TimeEval generates a parent directory named by `<algorithm_1>`, which contains a set of nested directories named by `<hyper_params_id>`. Following the directory tree, the deepest directory contains a set of files associated with respective dataset and algorithm as:

### docker-algorithm-scores.csv

This single-column file stores evaluation score for the each datapoint of the time series. 

### anomaly_scores.ts

Normalization of the `docker-algorthim-scores.csv` file yields the `anomaly_scores.ts` where every datapoint of the timeseries get a score between 0 to 1. For an anomaly, the score tends to 1. 

### model.pkl

For semi-supervised and supervised algorithm, a pickel file gets generated which contains a trained model from labelled datapoints.  

### execution.log

This log-report summarizes the execution process. Mainly used for debugging purpose as upon failure, it records the source of errors. It also stores the calculated metric score for the applied algorithm on the dataset.

### metrics.csv

   This file consisting information related to metrics used in the experiment. Find more information about it: `api/timeeval.metrics`

### hyper_params.json
This file stores the information about hyperparameters associated with the applied algorithm.
	
	
