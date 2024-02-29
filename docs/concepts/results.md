# TimeEval results

Upon executing TimeEval python script for the given datasets, TimeEval checks the effectiveness of selected algorithm and generate an output file as `./TimeEval/results/<timestamp>` directory. The output directory has following data structure:

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
|   └── dataset_2/epoch/
│       ├── anomaly_scores.ts
│       ├── docker-algorithm-scores.csv
│       ├── execution.log
|	├── model.pkl
│       ├── hyper_params.json
│       └── metrics.csv
└── algorithm_2/hyperparameter/
    └── dataset_1/epoch/
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

## Directory - alg_1
The directory contains a set of directories named by `<hyper_params_id>` used for algorithm. For every modification in hyperparameter, TimeEval will generate a new directory with new `<hyper_params_id>`. The `<hyper_params_id>` directory contains a set of directories named by input dataset. 

Result associated with respective dataset and algorithm is stored in following files:

### anomaly_scores.ts

### docker-algorithm-scores.csv

### model.pkl

### execution.log

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
	
	
	
