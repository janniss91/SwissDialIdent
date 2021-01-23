import csv
import numpy as np
import torch
from numpy import ndarray
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from typing import Tuple

DIALECT_MAP = {"LU": 0, "BE": 1, "ZH": 2, "BS": 3}


class SwissDialectDataset(Dataset):
    def __init__(self, ivectors: ndarray, labels: ndarray):
        """
        A dataset object that is supposed to store Swiss Dialect data for
        PyCharm models.

        :param ivectors: stores ivectors of all utterances with 400 values
         per sample
        :param labels: stores the dialect labels for each sample
        """
        self.ivectors = torch.from_numpy(ivectors)
        self.labels = torch.from_numpy(labels)
        self.n_samples = self.ivectors.shape[0]
        self.n_features = self.ivectors.shape[1]
        self.n_classes = len(DIALECT_MAP)

    def __getitem__(self, index: int) -> Tuple[ndarray, int]:
        return self.ivectors[index], self.labels[index]

    def __len__(self) -> int:
        return self.n_samples


def load_ivectors(vec_file: str) -> ndarray:
    """
    Load all ivectors from a space-separated file.

    :param vec_file: name of a space-separated file with i-vectors in it
    :return: a numpy array with all ivectors with shape (n_samples, 400)
    """
    with open(vec_file) as in_file:
        reader = csv.reader(in_file, delimiter=" ")
        ivectors = [[float(num) for num in ivec] for ivec in reader]

    return np.array(ivectors, dtype=np.float32)


def load_labels(txt_file: str) -> ndarray:
    """
    Load all labels from a text file that contains utterances in 4 Swiss German
    dialects with their respective dialect labels separated by a tab.

    :param txt_file: name of a text file with utterances and dialect labels
    :return: a numpy array with all labels with shape (n_samples, 1)
    """
    with open(txt_file) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        numeric_labels = [DIALECT_MAP[label] for _utterance, label in reader]

    return np.array(numeric_labels, dtype=np.int64)


def combine_data(
    train_vec_file: str,
    train_txt_file: str,
    dev_vec_file: str,
    dev_txt_file: str,
    shuffled: bool = True,
) -> Tuple[ndarray, ndarray]:
    """
    Combine train and test data so that they can be combined and shuffled for
    cross validation.

    :param train_vec_file: name of train file with i-vectors
    :param train_txt_file: name of train file with utterances and dialect labels
    :param test_vec_file: name of test file with i-vectors
    :param test_txt_file: name of test file with utterances and dialect labels
    :param shuffled: if True, the data is shuffled
    :return: a tuple of numpy arrays with all i-vectors and all labels
    """

    train_ivectors = load_ivectors(train_vec_file)
    dev_ivectors = load_ivectors(dev_vec_file)
    train_labels = load_labels(train_txt_file)
    dev_labels = load_labels(dev_txt_file)

    all_ivectors = np.concatenate((train_ivectors, dev_ivectors), axis=0)
    all_labels = np.concatenate((train_labels, dev_labels), axis=0)

    if shuffled:
        all_ivectors, all_labels = shuffle(all_ivectors, all_labels)

    return all_ivectors, all_labels
