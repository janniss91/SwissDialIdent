import time
from numpy import ndarray
from sklearn.svm import SVC
from typing import Tuple

from metrics import Metrics
from trainer import Trainer


class SVMTrainer(Trainer):
    def train(
        self,
        model: SVC,
        train_dataset: Tuple[ndarray, ndarray],
        test_dataset: Tuple[ndarray, ndarray],
    ):

        train_ivectors = train_dataset[0]
        train_labels = train_dataset[1]

        self.logger.train_samples = train_labels.shape[0]
        self.logger.test_samples = test_dataset[1].shape[0]

        # Track date, time and training time.
        train_time = time.strftime("%a-%d-%b-%Y-%H:%M:%S", time.localtime())
        start_time = time.time()

        # Training metrics will be printed directly from sklearn if verbose is true.
        model.fit(train_ivectors, train_labels)

        # Test and store test losses.
        metrics = self.test(model, test_dataset)

        end_time = time.time()
        runtime = round(end_time - start_time, 2)
        self.logger.log_metrics(train_time, runtime, metrics)

        return metrics

    def test(
        self, model: SVC, test_dataset: Tuple[ndarray, ndarray]
    ):
        test_ivectors = test_dataset[0]
        test_labels = test_dataset[1]

        predictions = model.predict(test_ivectors)

        metrics = Metrics(test_labels, predictions)

        return metrics
