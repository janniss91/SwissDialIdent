from numpy import ndarray
from sklearn.metrics import classification_report


class Metrics:
    def __init__(self, pred: ndarray, true: ndarray):
        self.pred = pred
        self.true = true
        self.label_names = ["LU", "BE", "ZH", "BS"]

        self.metrics_setup()

    def metrics_setup(self):
        metrics_dict = classification_report(self.true, self.pred, target_names=self.label_names, output_dict=True)
        for key in metrics_dict:
            setattr(self, key, metrics_dict[key])

    def print(self):
        print(classification_report(self.true, self.pred, target_names=self.label_names))

    def write_to_file(self):
        pass
