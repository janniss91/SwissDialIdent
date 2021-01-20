import time
import numpy as np
from numpy import ndarray
from sklearn.svm import SVC

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
        train_ivectors: ndarray,
        train_labels: ndarray,
        test_ivectors: ndarray = None,
        test_labels: ndarray = None,
        verbose: bool = False,
    ):

        self.logger.train_samples = train_labels.shape[0]
        self.logger.test_samples = test_labels.shape[0]

        train_time = time.strftime("%a-%d-%b-%Y-%H:%M:%S", time.localtime())
        start_time = time.time()

        verbose = 1 if verbose is True else 0
        model = SVC(
            C=self.c,
            kernel=self.kernel,
            degree=self.degree,
            max_iter=self.max_iter,
            verbose=verbose,
        )

        self.logger.model_name = model.__class__.__name__

        model.fit(train_ivectors, train_labels)
        metrics = self.test(model, test_ivectors, test_labels)

        end_time = time.time()
        runtime = round(end_time - start_time, 2)
        self.logger.log_metrics(train_time, runtime, metrics)

        return model, metrics

    def test(self, model: SVC, test_ivectors: ndarray, test_labels: ndarray):

        predictions = model.predict(test_ivectors)
        metrics = Metrics(test_labels, predictions)

        return metrics
