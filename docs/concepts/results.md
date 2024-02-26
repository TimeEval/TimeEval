# TimeEval results

Upon executing TimeEval python script for the given datasets, timeeval checks the effectiveness of selected algorithm and generate an output file as './TimeEval/results/<timestamp>' directory. The output directory has following data structure:

```bash
results_dir
├── results.csv
├── alg_1
|   ├── dataset_1
│   |   ├── anomaly_scores.ts
│   |   ├── docker-algorithm-scores.csv
│   |   ├── execution.log
│   |   ├── hyper_params.json
│   |   └── metrics.csv
|   └── dataset_2
│       ├── anomaly_scores.ts
│       ├── docker-algorithm-scores.csv
│       ├── execution.log
│       ├── hyper_params.json
│       └── metrics.csv
└── alg_2
    └── dataset_1
        ├── anomaly_scores.ts
        ├── docker-algorithm-scores.csv
        ├── execution.log
        ├── hyper_params.json
        └── metrics.csv

```

## File - result.csv
The file 'result.csv' contains following attributes:

- algorithm: applied on the dataset
- collection: indicates the name of the dataset 
- dataset: indicates the targeted segment of the data collection
- algo_training_type: three learning types: unsupervised, semi-supervised, supervised
- algo_input_dimensionality: two kinds: univarites, multivariate
- dataset_training_type: three learning type as above 
- dataset_input_dimensionality: two kinds as above
- train_preprocess_time: 
- train_main_time:
- execute_preprocess_time:
- execute_main_time: time taken for execute the script for the given datafile
- execute_postprocess_time: 
- status: assigns flag value upon the process
- error_message:
- repetition: number of times dataset being processed
- hyper_params: for given algorithm
- hyper_params_id: 32 alphanumerical characters
- ROC_AUC: varies between 0 and 1.
- FIXED_RANGE_PR_AUC: 

## Directory - alg_1
The directory contains a set of directories named by <hyper_params_id> for used algorithm. For every modification in hyperparameter, TimeEval will generate a new directory with new <hyper_params_id>. <hyper_params_id> directory contains a set of directories named by input dataset. 

Result associated with respective dataset and algorithm is stored in following files:

### anomaly_scores.ts

### docker-algorithm-scores.csv

### metrics.csv
   This file consisting information related to metrics used in the experiment. In the above syntax, metrics can be selected by defining metric list as [met1, met2, ...]. Most common metric being used in TimeEval are:
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
	
	
	
