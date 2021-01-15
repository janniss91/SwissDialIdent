import csv
import os
from typing import Dict
from typing import List


class TrainLogger:
    def __init__(self, model_type, n_epochs, batch_size, lr, log_interval):
        self.LOG_DIR = "train_logs"
        self.LOSS_LOG_DIR = "loss_logs"
        self.METRICS_FILE = "metric_logs.tsv"

        self.METRICS_HEADER = [
            "id",
            "datetime",
            "model_type",
            "runtime",
            "n_epochs",
            "batch_size",
            "lr",
            "LU-precision",
            "LU-recall",
            "LU-f1",
            "LU-support",
            "BE-precision",
            "BE-recall",
            "BE-f1",
            "BE-support",
            "ZH-precision",
            "ZH-recall",
            "ZH-f1",
            "ZH-support",
            "BS-precision",
            "BS-recall",
            "BS-f1",
            "BS-support",
            "accuracy",
            "macro-avg-precision",
            "macro-avg-recall",
            "macro-avg-f1",
            "macro-avg-support",
            "weighted-avg-precision",
            "weighted-avg-recall",
            "weighted-avg-f1",
            "weighted-avg-support",
            # ...
            # The "comments" section is not filled automatically but can be
            # used to manually comment on peculiarities about the training run.
            "comments",
        ]

        self.model_type = model_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval

        self.model_name = self.model_type.__class__.__name__

    def log_metrics(self, train_time: str, runtime: float, metrics: Dict):

        metrics_path = os.path.join(self.LOG_DIR, self.METRICS_FILE)
        with open(metrics_path, "r+") as metrics_file:

            metrics_writer = csv.writer(metrics_file, delimiter="\t")

            # If the file is completely empty, a header line is added.
            if os.path.getsize() == 0:
                metrics_writer.write(self.METRICS_HEADER)

            # Choose the ID for the training run.
            id_reader = csv.reader(metrics_file, delimiter="\t")
            ids = [line[0] for line in id_reader][-1]
            next_id = ids[-1] + 1

            train_info = [
                next_id,
                train_time,
                self.model_name,
                runtime,
                self.n_epochs,
                self.batch_size,
                self.lr,
                metrics["LU"]["precision"],
                metrics["LU"]["recall"],
                metrics["LU"]["f1-score"],
                metrics["LU"]["support"],
                metrics["BE"]["precision"],
                metrics["BE"]["recall"],
                metrics["BE"]["f1-score"],
                metrics["BE"]["support"],
                metrics["ZH"]["precision"],
                metrics["ZH"]["recall"],
                metrics["ZH"]["f1-score"],
                metrics["ZH"]["support"],
                metrics["BS"]["precision"],
                metrics["BS"]["recall"],
                metrics["BS"]["f1-score"],
                metrics["BS"]["support"],
                metrics["accuracy"],
                metrics["macro avg"]["precision"],
                metrics["macro avg"]["recall"],
                metrics["macro avg"]["f1-score"],
                metrics["macro avg"]["support"],
                metrics["weighted avg"]["precision"],
                metrics["weighted avg"]["recall"],
                metrics["weighted avg"]["f1-score"],
                metrics["weighted avg"]["support"],
                "",
            ]

            metrics_writer.write(train_info)

    def log_losses(self, train_time: str, metrics: List):
        pass

    def __call__(self, train_time: str, runtime: float, metrics: Dict, losses: List):
        self.log_metrics(train_time, runtime, metrics)
        self.log_losses(train_time, runtime, losses)