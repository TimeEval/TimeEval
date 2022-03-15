from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_novelty_svr_parameters: Dict[str, Dict[str, Any]] = {
 "C": {
  "defaultValue": 1.0,
  "description": "Online SVR parameter: Penalty parameter C of the error term.",
  "name": "C",
  "type": "float"
 },
 "anomaly_window_size": {
  "defaultValue": 6,
  "description": "Size of event windows, also called event duration, for which suprising occurences are aggregated. Should not be chosen too large! (paper: n)",
  "name": "anomaly_window_size",
  "type": "int"
 },
 "coef0": {
  "defaultValue": 0.0,
  "description": "Online SVR parameter: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.",
  "name": "coef0",
  "type": "float"
 },
 "degree": {
  "defaultValue": 3,
  "description": "Online SVR parameter: Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.",
  "name": "degree",
  "type": "int"
 },
 "epsilon": {
  "defaultValue": 0.1,
  "description": "Specifies epsilon-tube to find suprising occurences in the prediction residuals (resid !> 2eps). Reused as Online SVR parameter: Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.",
  "name": "epsilon",
  "type": "float"
 },
 "forgetting_time": {
  "defaultValue": None,
  "description": "If this is set, points older than forgetting_time are removed from the model (forgotten) (paper: W)",
  "name": "forgetting_time",
  "type": "int"
 },
 "gamma": {
  "defaultValue": None,
  "description": "Online SVR parameter: Kernel coefficient for 'poly', 'sigmoid', and 'rbf'-kernels. If gamma is None then 1/n_features will be used instead.",
  "name": "gamma",
  "type": "float"
 },
 "kernel": {
  "defaultValue": "rbf",
  "description": "Online SVR parameter: Specifies the kernel type to be used in the algorithm.",
  "name": "kernel",
  "type": "enum[linear,poly,rbf,sigmoid,rbf-gaussian,rbf-exp]"
 },
 "lower_suprise_bound": {
  "defaultValue": None,
  "description": "Number of suprising occurences that must be present within an event (see window_size) to regard the event as novel/anomalous (paper: h). Range: 0 < lower_suprise_bound < window_size. If not supplied 'h = window_size / 2' is used as default.",
  "name": "lower_suprise_bound",
  "type": "int"
 },
 "n_init_train": {
  "defaultValue": 500,
  "description": "Number of initial points to fit regression model on. For those points no score is calculated.",
  "name": "n_init_train",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "scaling": {
  "defaultValue": "standard",
  "description": "If the data should be scaled/normalized before regression using StandardScaler, RobustScaler, or PowerTransformer (Yeo-Johnson + standard scaling). See https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py.",
  "name": "scaling",
  "type": "enum[None,standard,robust,power]"
 },
 "stabilized": {
  "defaultValue": "True",
  "description": "Online SVR parameter: If stabilization should be used.",
  "name": "stabilized",
  "type": "boolean"
 },
 "tol": {
  "defaultValue": 0.001,
  "description": "Online SVR parameter: Tolerance for stopping criterion.",
  "name": "tol",
  "type": "float"
 },
 "train_window_size": {
  "defaultValue": 16,
  "description": "Size of training windows, also called embedding dimensions, used as context to predict the next point (paper: D)",
  "name": "train_window_size",
  "type": "int"
 },
 "verbose": {
  "defaultValue": 0,
  "description": "Controls verbose output. Higher values mean more detailled output [0; 5]. Verbose output of the Online SVR appears not until >=3.",
  "name": "verbose",
  "type": "int"
 }
}


def novelty_svr(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="NoveltySVR",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/novelty_svr",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_novelty_svr_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
