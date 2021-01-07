from numpy import ndarray
from sklearn.metrics import classification_report


class Metrics:
    def __init__(self, true: ndarray, pred: ndarray):
        self.true = true
        self.pred = pred
        self.label_names = ["LU", "BE", "ZH", "BS"]

        self.set_up_metrics()

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = []

    def set_up_metrics(self):
        metrics_dict = classification_report(
            self.true, self.pred, target_names=self.label_names, output_dict=True
        )
        for key in metrics_dict:
            setattr(self, key, metrics_dict[key])

    def store_losses(self, train_losses, train_counter, test_losses, test_counter):
        self.train_losses = train_losses
        self.train_counter = train_counter
        self.test_losses = test_losses
        self.test_counter = test_counter

    def print(self):
        print(
            classification_report(self.true, self.pred, target_names=self.label_names)
        )

    def write_to_file(self):
        pass
