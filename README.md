# Swiss Dialect Identification

In this project I compare different machine learning setups for the identification of 4 Swiss German dialects.

## Setup

For the setup of this repository simply type:

    make

This will

- set up a virtual environment for this repository,
- install all necessary project dependencies.

If anything does not work, try installing with `Python 3.8`.

## Clean and Re-install

To reset the repository to its inital state, type:

    make dist-clean

This will remove the virtual environment and all dependencies.  
With the `make` command you can re-install them.

To remove temporary files like .pyc or .pyo files, type:

    make clean

## Data

The data for this project comes from 3rd VarDial Evaluation Campaign from 2019 (for more information see: https://sites.google.com/view/vardial2019/campaign).

It consists of a training and a validation set.  
The data comprises utterances from 4 Swiss German dialects (Basel, Bern, Lucerne, Zurich). The dataset comes with three different representations of the utterances.

- The files `train.txt` and `dev.txt` contain an utterance per line combined with its dialect label. The label is separated from the utterance by a tab.
- The files `train.vec` and `dev.vec` contain a so-called iVector per line. The iVector is a vector of floating-point numbers, which represents the spoken form of the utterances. The iVector has been extracted from audio files.
- (The original dataset contained another file for training and test set that kept the normalized forms of the utterances. This data is not needed for this project and cannot be found in the repository.)

The training set contains 14279 utterances and the development set contains 4530 utterances.

## Models and Training

### Models

- Multinomial Logistic Regression (Baseline Model)
- SVM

### Training

It is possible to train a single model (-s), perform cross validation (-v) to select a best model or train a final model (-f) on the whole data set.  
Besides, there is the chance to use the original train-test split or shuffle the data (-o).
Eventually, you can choose to train with print output or not (-v).

Examples:

Train a single SVM model with original split:

    python3 scripts/train.py -s SVM -o

Train a single logistic regression model with randomized split (implementation still missing):

    python3 scripts/train.py -s LogisticRegression

Choose a best model with cross validation:

    python3 scripts/train.py -c

Train the best model (here assumed it is an SVM) on the whole dataset (train and dev):

    python3 scripts/train.py -f SVM

NOTE: Only a single model (not final) can be trained on the original split.

### Storing a Model

Models can be stored in the repository.  
At the end of each training run you are asked whether you want to store the output model or not.

Models are stored in the directory `stored_models`.

## Configuration Files

The functionality is not implemented yet but the aim is to store training parameters in configuration files. There will also be a script to generate configuration files.

## Logging

Train losses and metrics are stored in the directory `train_logs`.  
All information about each training run can be found here.
