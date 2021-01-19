import time
from numpy import ndarray
from sklearn.svm import SVC
from typing import Tuple

from metrics import Metrics
from trainer import Trainer
from train_logger import TrainLogger


class SVMTrainer(Trainer):
    def __init__(self, c=1.0, kernel="rbf", degree=3, max_iter=-1):
        self.c = 1.0
        self.kernel = kernel
        self.degree = degree
        self.max_iter = max_iter

        self.logger = TrainLogger()
        for attribute in self.__dict__:
            setattr(self.logger, attribute, self.__dict__[attribute])

        # It is important that the super initialization happens after
        # setting the trainlogger attributes.
        super(SVMTrainer, self).__init__()

    def train(
        self,
        train_dataset: Tuple[ndarray, ndarray],
        test_dataset: Tuple[ndarray, ndarray],
        verbose: bool = False,
    ):

        train_ivectors = train_dataset[0]
        train_labels = train_dataset[1]

        self.logger.train_samples = train_labels.shape[0]
        self.logger.test_samples = test_dataset[1].shape[0]

        # Track date, time and training time.
        train_time = time.strftime("%a-%d-%b-%Y-%H:%M:%S", time.localtime())
        start_time = time.time()

        model = SVC(
            C=self.c,
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            verbose=verbose,
        )

        self.logger.model_name = model.__class__.__name__

        # Training metrics will be printed directly from sklearn if verbose is true.
        model.fit(train_ivectors, train_labels)
        metrics = self.test(model, test_dataset)

        end_time = time.time()
        runtime = round(end_time - start_time, 2)
        self.logger
        self.logger.log_metrics(train_time, runtime, metrics)

        return metrics

    def test(self, model: SVC, test_dataset: Tuple[ndarray, ndarray]):
        test_ivectors = test_dataset[0]
        test_labels = test_dataset[1]

        predictions = model.predict(test_ivectors)

        metrics = Metrics(test_labels, predictions)

        return metrics
