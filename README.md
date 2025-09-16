# WorkingMemoryCapacity

## Data

The data.zip file comprises the parent folder, <i>data</i>, under which there exist three separate folders each denoting a separate experiment, namely <i>DnB1, DnB1+DnB2, DnB2</i>. 

Each of these folders consist of 5 sub-folders denoting the number of chunks used in the experiments. In each sub-folder, there exists the train and test folders for training and testing our models, respectively.

The target.xlsx file contains the target label for each participant.

All files have been anonymized.

## Code

We provide code required to reproduce our randomforest model experiments, including cross-validation, training and testing, feature importance and counterfactual permutation ablation.

For reproducability, run the following commands:

`python machine_learning_training_rf.py`

`python ablation_study.py`
