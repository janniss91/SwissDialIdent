from numpy import ndarray
from sklearn.metrics import classification_report
from typing import List


class Metrics:
    def __init__(self, true: ndarray, pred: ndarray):
        """
        A class to store all metrics (and losses) that result from a training
        run.
        The Metrics class is used as an instance variable of the Trainer class.

        :param true: All actual labels of the  test set.
        :param pred: All predicted labels of the test set.
        """
        self.true = true
        self.pred = pred
        # The order of the labels MUST REMAIN this way!
        self.label_names = ["LU", "BE", "ZH", "BS"]

        self.set_up_metrics()

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

    def set_up_metrics(self):
        """
        On the basis of the true and predicted values, the values of the
        sklearn classification report are stored in the Metrics object.
        """
        metrics_dict = classification_report(
            self.true, self.pred, target_names=self.label_names, output_dict=True
        )
        # Make sure no attribute contains spaces.
        metrics_dict["macroavg"] = metrics_dict.pop("macro avg")
        metrics_dict["weightedavg"] = metrics_dict.pop("weighted avg")

        for key in metrics_dict:
            setattr(self, key, metrics_dict[key])

    def store_losses(
        self,
        train_losses: List[float],
        train_counter: List[int],
        test_losses: List[float],
        test_counter: List[int],
    ):
        """
        The losses that come up during the test runs are stored in the Metrics
        object, too.
        This does not apply to SVM classification so far.

        :param train_losses: All losses that come up during training runs.
        :param train_counter: Counts that show the number of samples to make it
         possible to map them to the respective train losses.
        :param test_losses: Losses that come up during the test run.
        :param train_counter: Counts that show the number of samples to make it
         possible to map them to the respective test losses.
        """
        self.train_losses = train_losses
        self.train_counter = train_counter
        self.test_losses = test_losses
        self.test_counter = test_counter

    def print(self):
        """
        Print all relevant metrics to stdout.
        """
        print(
            classification_report(self.true, self.pred, target_names=self.label_names)
        )
