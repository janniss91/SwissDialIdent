import numpy as np
from sklearn.model_selection import KFold

from dataset import SwissDialectDataset


class Trainer:
    def __init__(
        self,
        model_type,  # DataType: [LogisticRegression, ...]
        ivectors: np.ndarray,
        labels: np.ndarray,
        n_epochs=10,
        batch_size=10,
        lr=0.01,
        logging_interval=50
    ):
        self.model_type = model_type
        self.ivectors = ivectors
        self.labels = labels
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.logging_interval = logging_interval

        self.models_and_metrics = []

    def train(self):
        raise NotImplementedError(
            "The train method can only be run from a subclass of Trainer."
        )

    def test(self):
        raise NotImplementedError(
            "The test method can only be run from a subclass of Trainer."
        )

    def cross_validation(self, k=10, verbose=False):
        kfold = KFold(k)
        for k, (train_ids, test_ids) in enumerate(kfold.split(self.labels), start=1):

            train_ivecs = self.ivectors[train_ids]
            train_labels = self.labels[train_ids]
            test_ivecs = self.ivectors[test_ids]
            test_labels = self.labels[test_ids]

            train_dataset = SwissDialectDataset(train_ivecs, train_labels)
            test_dataset = SwissDialectDataset(test_ivecs, test_labels)

            input_dim = train_dataset.n_features
            output_dim = train_dataset.n_classes

            model = self.model_type(input_dim, output_dim)

            if verbose:
                print("K-Fold Cross validation: k=" + str(k))

            # Test the performance on the dev set with randomly initialized weights.
            # This way it can be compared to the performance after training.
            self.test(model, test_dataset, verbose=verbose)

            model, train_losses, train_counter, test_losses, test_counter = self.train(
                model, train_dataset, test_dataset, verbose
            )
