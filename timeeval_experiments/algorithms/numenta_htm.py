from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_numenta_htm_parameters: Dict[str, Dict[str, Any]] = {
 "activationThreshold": {
  "defaultValue": 12,
  "description": "Segment activation threshold. A segment is active if it has >= tpSegmentActivationThreshold connected synapses that are active due to infActiveState",
  "name": "activationThreshold",
  "type": "int"
 },
 "alpha": {
  "defaultValue": 0.5,
  "description": "This controls how fast the classifier learns/forgets. Higher values make it adapt faster and forget older patterns faster",
  "name": "alpha",
  "type": "float"
 },
 "autoDetectWaitRecords": {
  "defaultValue": 50,
  "description": "",
  "name": "autoDetectWaitRecords",
  "type": "int"
 },
 "cellsPerColumn": {
  "defaultValue": 32,
  "description": "The number of cells (i.e., states), allocated per column.",
  "name": "cellsPerColumn",
  "type": "int"
 },
 "columnCount": {
  "defaultValue": 2048,
  "description": "Number of cell columns in the cortical region (same number for SP and TM)",
  "name": "columnCount",
  "type": "int"
 },
 "encoding_input_width": {
  "defaultValue": 21,
  "description": "",
  "name": "encoding_input_width",
  "type": "int"
 },
 "encoding_output_width": {
  "defaultValue": 50,
  "description": "",
  "name": "encoding_output_width",
  "type": "int"
 },
 "globalDecay": {
  "defaultValue": 0.0,
  "description": "",
  "name": "globalDecay",
  "type": "float"
 },
 "initialPerm": {
  "defaultValue": 0.21,
  "description": "Initial Permanence",
  "name": "initialPerm",
  "type": "float"
 },
 "inputWidth": {
  "defaultValue": 2048,
  "description": "",
  "name": "inputWidth",
  "type": "int"
 },
 "maxAge": {
  "defaultValue": 0,
  "description": "",
  "name": "maxAge",
  "type": "int"
 },
 "maxSegmentsPerCell": {
  "defaultValue": 128,
  "description": "Maximum number of segments per cell",
  "name": "maxSegmentsPerCell",
  "type": "int"
 },
 "maxSynapsesPerSegment": {
  "defaultValue": 32,
  "description": "Maximum number of synapses per segment",
  "name": "maxSynapsesPerSegment",
  "type": "int"
 },
 "minThreshold": {
  "defaultValue": 9,
  "description": "Minimum number of active synapses for a segment to be considered during search for the best-matching segments.",
  "name": "minThreshold",
  "type": "int"
 },
 "newSynapseCount": {
  "defaultValue": 20,
  "description": "New Synapse formation count",
  "name": "newSynapseCount",
  "type": "int"
 },
 "numActiveColumnsPerInhArea": {
  "defaultValue": 40,
  "description": "Maximum number of active columns in the SP region's output (when there are more, the weaker ones are suppressed)",
  "name": "numActiveColumnsPerInhArea",
  "type": "int"
 },
 "pamLength": {
  "defaultValue": 1,
  "description": "\"Pay Attention Mode\" length. This tells the TM how many new elements to append to the end of a learned sequence at a time. Smaller values are better for datasets with short sequences, higher values are better for datasets with long sequences.",
  "name": "pamLength",
  "type": "int"
 },
 "permanenceDec": {
  "defaultValue": 0.1,
  "description": "Permanence Decrement",
  "name": "permanenceDec",
  "type": "float"
 },
 "permanenceInc": {
  "defaultValue": 0.1,
  "description": "Permanence Increment",
  "name": "permanenceInc",
  "type": "float"
 },
 "potentialPct": {
  "defaultValue": 0.5,
  "description": "What percent of the columns's receptive field is available for potential synapses. At initialization time, we will choose potentialPct * (2*potentialRadius+1)^2",
  "name": "potentialPct",
  "type": "float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
 "synPermActiveInc": {
  "defaultValue": 0.1,
  "description": "",
  "name": "synPermActiveInc",
  "type": "float"
 },
 "synPermConnected": {
  "defaultValue": 0.1,
  "description": "The default connected threshold. Any synapse whose permanence value is above the connected threshold is a \"connected synapse\", meaning it can contribute to the cell's firing. Typical value is 0.10. Cells whose activity level before inhibition falls below minDutyCycleBeforeInh will have their own internal synPermConnectedCell threshold set below this default value.",
  "name": "synPermConnected",
  "type": "float"
 },
 "synPermInactiveDec": {
  "defaultValue": 0.005,
  "description": "",
  "name": "synPermInactiveDec",
  "type": "float"
 }
}


def numenta_htm(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="NumentaHTM",
        main=DockerAdapter(
            image_name="sopedu:5000/akita/numenta_htm",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_numenta_htm_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("univariate")
    )
