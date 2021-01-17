from numpy import ndarray
from torch import Tensor
from sklearn.model_selection import KFold

from dataset import SwissDialectDataset
from train_logger import TrainLogger
from metrics import Metrics


class Trainer:
    def __init__(
        self,
        model_type,  # DataType: [LogisticRegression, ...]
        ivectors: ndarray,
        labels: ndarray,
        n_epochs: int = 10,
        batch_size: int = 10,
        lr: float = 0.01,
        log_interval: int = 50,
    ):
        self.model_type = model_type
        self.ivectors = ivectors
        self.labels = labels
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval

        self.cv_metrics = []
        self.logger = TrainLogger(
            self.model_type,
            self.n_epochs,
            self.batch_size,
            self.lr,
            self.log_interval,
        )

    def train(self):
        raise NotImplementedError(
            "The train method can only be run from a subclass of Trainer."
        )

    def test(self):
        raise NotImplementedError(
            "The test method can only be run from a subclass of Trainer."
        )

    def cross_validation(self, k: int = 10, verbose: bool = False):
        kfold = KFold(k)
        for k, (train_ids, test_ids) in enumerate(kfold.split(self.labels), start=1):

            if verbose:
                print("K-Fold Cross validation: k=" + str(k))

            train_ivecs = self.ivectors[train_ids]
            train_labels = self.labels[train_ids]
            test_ivecs = self.ivectors[test_ids]
            test_labels = self.labels[test_ids]

            # PyTorch models (up to now only LogisticRegression).
            if self.model_type.__name__ in ("LogisticRegression"):
                train_dataset = SwissDialectDataset(train_ivecs, train_labels)
                test_dataset = SwissDialectDataset(test_ivecs, test_labels)
                input_dim = train_dataset.n_features
                output_dim = train_dataset.n_classes
                model = self.model_type(input_dim, output_dim)

                # Test the performance on the dev set with randomly initialized
                # weights. This way it can be compared to the performance after
                # training.
                self.test(model, test_dataset, verbose=verbose)
                metrics = self.train(model, train_dataset, test_dataset, verbose)

            # Up to now only case for SVM.
            else:
                train_dataset = (train_ivecs, train_labels)
                test_dataset = (test_ivecs, test_labels)
                # Todo: The model might need more hyperparameters.
                model = self.model_type(verbose=verbose)
                metrics = self.train(model, train_dataset, test_dataset)

            self.cv_metrics.append((self.model_type.__name__ + "-split-" + str(k), metrics))

# Todo: Test that the numbers are printed correctly!
    def print_train_metrics(
        self,
        epoch: int,
        batch_id: int,
        n_samples: int,
        loss: Tensor,
    ):
        batch_count = batch_id * self.batch_size

        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_count,
                n_samples,
                100.0 * batch_count / n_samples,
                loss.item(),
            )
        )

    def print_test_metrics(
        self,
        test_loss: float,
        correct: Tensor,
        n_samples: int,
        metrics: Metrics,
    ):
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, n_samples, 100.0 * metrics.accuracy
            )
        )

        metrics.print()
