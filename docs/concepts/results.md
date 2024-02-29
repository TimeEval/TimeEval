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

For a given dataset, different algorithms with varying hyperparameters would yield distict results. The file `result.csv` provides an overview of evalution and contains following attributes:

| Column Name | Datatype | Description |
| --- | --- | --- |
| algorithm| str | applied on the dataset |
| collection| str | indicates the name of the dataset | 
| dataset| str | indicates the targeted segment of the data collection |
| algo_training_type | str | three learning types: unsupervised, semi-supervised, supervised |
| algo_input_dimensionality | str | two kinds: univarites, multivariate |
| dataset_training_type | str | three learning type as above | 
| dataset_input_dimensionality | str | two kinds as above |
| train_preprocess_time| float64| |
| train_main_time| float64 | |
| execute_preprocess_time| float64 | |
| execute_main_time | float64 | time taken for execute the script for the given datafile |
| execute_postprocess_time|  float64 | |
| status| str | assigns flag value upon the process |
| error_message| str | |
| repetition| int | number of times dataset being processed |
| hyper_params| float64 | for given algorithm |
| hyper_params_id| float64 | 32 alphanumerical characters |
| metric_1| float64 | --user-- |
| metric_2|  float64 | --user-- |

## Directory - <algorithm_1>/<hyper_params_id>/<collection_name>/<dataset_name_1>/<repetition_number>/
For every modification in hyperparameter of algorithms, TimeEval generates a parent directory named by `<algorithm_1>`, which contains a set of nested directories named by `<hyper_params_id>`. Following the directory tree, the base directory contains the following files. 

Result associated with respective dataset and algorithm is stored in following files:

### docker-algorithm-scores.csv
This single-column file stores evaluation score for the each datapoint of the time series. 

### anomaly_scores.ts
Normalization of the `docker-algorthim-scores.csv` file yields the `anamoly_scores.ts` where every datapoint of the timeseries get a score between 0 to 1. For an anamoly, the score tends to 1. 

### model.pkl
For semi-supervised and supervised algorithm, a pickel file required which contains the a trained model from a labelled dataset.  

### execution.log
This log report summarizes the execution process. It provides information about the algorithm type and saves the calculated metric score for the applied algorithm on the dataset.

### metrics.csv
   This file consisting information related to metrics used in the experiment. In the above syntax, metrics can be selected by defining metric list as `[met1, met2, ...]`. Most common metric being used in TimeEval are:
1. Classification-metrics: for binary dataset
   - Precision
   - Recall
   - F1Score
2. AUC-metrics: for continuous dataset
   - RocAUC
   - PrAUC
4. Range-metrics: analyse segment of the dataset
   - RangePrecision
   - RangeRecall
   - RangeFScore
   - RangePrecisionRangeRecallAUC

### hyper_params.json
This file stores the information about hyperparameters associated with the applied algorithm.
	
	
