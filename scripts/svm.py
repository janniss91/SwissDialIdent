import time
from numpy import ndarray
from sklearn.svm import SVC

from metrics import Metrics
from trainer import Trainer
from train_logger import TrainLogger


class SVMTrainer(Trainer):
    def __init__(self, c=1.0, kernel="rbf", degree=3, max_iter=-1):
        """
        Inherits from Trainer class.
        This class handles the training and test processes of the SVM
        model.

        :param c: the penalty for all misclassified data samples in the model;
        the penalty for a datapoint is proportional to the distance of the
        point to the decision boundary
        :param kernel: function to transform the data in the desired way
        :param degree: Degree of the polynomial kernel function (‘poly’);
        ignored by all other kernel functions
        :param max_iter: limit of iterations; no limit if it is -1
        """
        self.c = 1.0
        # TODO: Include gamma as parameter. High gamma: more likely to overfit
        # gamma only applicable to non-linear kernels
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
        """
        Train the SVM model.
        The test run is embedded here.

        :param train_ivecs: the array with all training i-vectors
        :param train_labels: the array with all training labels
        :param test_ivecs: the array with all test i-vectors
        :param test_labels: the array with all tesst labels
        :param verbose: if true, losses and metrics are printed during training
        :return: the trained model and the metrics from the last epoch
        """
        self.logger.train_samples = train_labels.shape[0]
        self.logger.test_samples = test_labels.shape[0]

        train_time = time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime())
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
        """
        Test the SVM model on the training set.

        :param model: the model that has been trained before
        :param test_ivecs: the array with all test i-vectors
        :param test_labels: the array with all tesst labels
        :return: the test loss and the metrics
        """
        predictions = model.predict(test_ivectors)
        metrics = Metrics(test_labels, predictions)

        return metrics

    def train_final_model(self, ivectors: ndarray, labels: ndarray, verbose: bool = True):
        """
        This method will only have to be implemented if the actual test data 
        can be used.
        """
        pass
