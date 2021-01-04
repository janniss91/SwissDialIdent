# Swiss Dialect Identification

## Setup

For the setup of this repository simply type:

    make

This will

- set up a virtual environment for this repository,
- install all necessary project dependencies.

If anything does not work, try installing with ```Python 3.8```.

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

- The files ```train.txt``` and ```dev.txt``` contain an utterance per line combined with its dialect label. The label is separated from the utterance by a tab.
- The files ```train.vec``` and ```dev.vec``` contain a so-called iVector per line. The iVector is a vector of floating-point numbers, which represents the spoken form of the utterances. The iVector has been extracted from audio files.
- (The original dataset contained another file for training and test set that kept the normalized forms of the utterances. This data is not needed for this project and cannot be found in the repository.)

The training set contains 14279 utterances and the development set contains 4530 utterances.

## Models

### Multinomial Logistic Regression (Baseline Model)

The Logistic Regression model serves as a baseline for the more advanced approaches.

Command for training the Logistic Regression Model:

    python3 scripts/logistic_regression.py

**Other Models will be set up in the future.**
