from numpy import ndarray
from torch import Tensor
from sklearn.model_selection import KFold

from train_logger import TrainLogger
from metrics import Metrics


class Trainer:
    def __init__(
        self,
        model_type,  # DataType: [LogisticRegression, ...]
        n_epochs: int = 10,
        batch_size: int = 10,
        lr: float = 0.01,
        log_interval: int = 50,
    ):
        self.model_type = model_type
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

    def cross_validation(
        self,
        ivectors: ndarray,
        labels: ndarray,
        k: int = 10,
        verbose: bool = False
    ):
        kfold = KFold(k)
        for k, (train_ids, test_ids) in enumerate(kfold.split(labels), start=1):

            train_ivecs = ivectors[train_ids]
            train_labels = labels[train_ids]
            test_ivecs = ivectors[test_ids]
            test_labels = labels[test_ids]

            if verbose:
                print("K-Fold Cross validation: k=" + str(k))

            model, metrics = self.train(self.model_type, train_ivecs, train_labels, test_ivecs, test_labels, verbose)

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
