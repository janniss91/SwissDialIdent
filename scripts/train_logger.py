import csv
import os
from typing import Dict


class TrainLogger:
    def __init__(self):

        # Output directories.
        self.LOG_DIR = "train_logs"
        self.LOSS_LOG_DIR = "loss_logs"
        self.METRICS_FILE = "metric_logs.tsv"

        self.METRICS_HEADER = (
            "id",
            "datetime",
            "runtime",
            "model_type",
            "n_samples",
            "train_samples",
            "test_samples",
            "epoch_no",
            "n_epochs",
            "batch_size",
            "lr",
            "c"
            "kernel"
            "degree"
            "max_iter"
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
            # The "comments" section is not filled automatically but can be
            # used to manually comment on peculiarities about the training run.
            "comments",
        )

        self.model_type = "-"

        # Parameters for PyTorch models.
        self.n_epochs = "-"
        self.batch_size = "-"
        self.lr = "-"
        self.log_interval = "-"
        self.epoch_no = "-"

        # Parameters for Support Vector Machine.
        self.c = "-"
        self.kernel = "-"
        self.degree = "-"
        self.max_iter = "-"

        self.train_samples = "-"
        self.test_samples = "-"

    def log_metrics(self, train_time: str, runtime: float, metrics: Dict):

        metrics_path = os.path.join(self.LOG_DIR, self.METRICS_FILE)

        with open(metrics_path, "r") as metrics_file:
            # Choose the ID for the training run.
            id_reader = csv.reader(metrics_file, delimiter="\t")
            ids = [line[0] for line in id_reader]
            next_id = 1 if len(ids) <= 1 else int(ids[-1]) + 1

        with open(metrics_path, "a") as metrics_file:

            metrics_writer = csv.writer(metrics_file, delimiter="\t")

            # If the file is completely empty, a header line is added.
            if os.path.getsize(metrics_path) == 0:
                metrics_writer.writerow(self.METRICS_HEADER)

            train_info = [
                next_id,
                train_time,
                str(runtime) + " sec",
                self.model_name,
                self.train_samples + self.test_samples,
                self.train_samples,
                self.test_samples,
                self.epoch_no,
                self.n_epochs,
                self.batch_size,
                self.lr,
                self.c,
                self.kernel,
                self.degree,
                self.max_iter,
            ]

            # Round all metrics to 3 digits before adding them.
            train_info.extend(
                [
                    round(value, 3) if isinstance(value, float) else value
                    for value in (
                        metrics.LU["precision"],
                        metrics.LU["recall"],
                        metrics.LU["f1-score"],
                        metrics.LU["support"],
                        metrics.BE["precision"],
                        metrics.BE["recall"],
                        metrics.BE["f1-score"],
                        metrics.BE["support"],
                        metrics.ZH["precision"],
                        metrics.ZH["recall"],
                        metrics.ZH["f1-score"],
                        metrics.ZH["support"],
                        metrics.BS["precision"],
                        metrics.BS["recall"],
                        metrics.BS["f1-score"],
                        metrics.BS["support"],
                        metrics.accuracy,
                        metrics.macroavg["precision"],
                        metrics.macroavg["recall"],
                        metrics.macroavg["f1-score"],
                        metrics.macroavg["support"],
                        metrics.weightedavg["precision"],
                        metrics.weightedavg["recall"],
                        metrics.weightedavg["f1-score"],
                        metrics.weightedavg["support"],
                        "-",
                    )
                ]
            )

            metrics_writer.writerow(train_info)

    def log_losses(self, train_time: str, metrics: Dict):
        train_counter = metrics.train_counter
        train_losses = metrics.train_losses
        test_counter = metrics.test_counter
        test_losses = metrics.test_losses

        loss_path = os.path.join(
            self.LOG_DIR, self.LOSS_LOG_DIR, train_time + "-" + self.model_name
        )
        with open(loss_path, "w") as loss_file:
            loss_writer = csv.writer(loss_file, delimiter="\t")

            loss_writer.writerow(("train_loss"))
            for (epoch, batch_count), loss in zip(train_counter, train_losses):
                loss_writer.writerow((epoch, batch_count, loss))
            loss_writer.writerow("")

            loss_writer.writerow(("test_loss"))
            for batch_count, loss in zip(test_counter, test_losses):
                loss_writer.writerow((batch_count, loss))
