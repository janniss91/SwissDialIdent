# Swiss Dialect Identification

## Data

The data for this project comes from 3rd VarDial Evaluation Campaign from 2019 (for more information see: https://sites.google.com/view/vardial2019/campaign).

It consists of a training and a validation set.  
The data comprises utterances from 4 Swiss German dialects (Basel, Bern, Lucerne, Zurich). The dataset comes with three different representations of the utterances.

- The files ```train.txt``` and ```dev.txt``` contain an utterance per line combined with its dialect label. The label is separated from the utterance by a tab.
- The files ```train.vec``` and ```dev.vec``` contain a so-called iVector per line. The iVector is a vector of floating-point numbers, which represents the spoken form of the utterances. The iVector has been extracted from audio files.
- (The original dataset contained another file for training and test set that kept the normalized forms of the utterances. This data is not needed for this project )

The training set contains 14279 utterances and the development set contains 4530 utterances.

## Models

### Multinomial Logistic Regression (Baseline Model)

The Logistic Regression model serves as a baseline for the more advanced approaches.

### SVM

The SVM is a traditional machine learning approach for classification.

### Gradient Boosting Model

Gradient Boosting is a traditional machine learning approach for classification.

### Recurrent Neural Network

Recurrent Neural Networks are a deep learning architecture that is particularly suitable for the classification of sequences. It is used for language processing a lot due to the sequential nature of language. The model is expected to outperform the traditional approaches.