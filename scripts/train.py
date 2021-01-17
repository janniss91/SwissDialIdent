from sklearn.svm import SVC
from typing import List
from typing import Tuple

from dataset import combine_data
from logistic_regression import LogisticRegression
from logistic_regression import LogisticRegressionTrainer
from metrics import Metrics
from svm import SVMTrainer


def select_model(
    ivectors,
    labels,
    # Todo: Check if these metrics should be different for different models.
    # Maybe use lists of batch sizes and learning rates (e.g.).
    n_epochs: int = 10,
    batch_size: int = 10,
    lr: int = 0.01,
    log_interval: int = 50,
    k: int = 10,
    verbose: bool = False,
):

    metrics_all_models = []
    model_types = [LogisticRegression, SVC]
    trainer_types = [LogisticRegressionTrainer, SVMTrainer]

    for model_type, trainer_type in zip(model_types, trainer_types):
        trainer = trainer_type(
            model_type,
            all_ivectors,
            all_labels,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            log_interval=log_interval,
        )
        trainer.cross_validation(verbose=True, k=k)
        metrics_all_models.append(trainer.cv_metrics)

    best_model = compare_metrics(metrics_all_models)

    return best_model


def compare_metrics(metrics_all_models: List[List[Tuple[str, Metrics]]]):
    # Todo: Put the metrics of all CVs from all models in here and
    pass


def train_best_model():
    pass


if __name__ == "__main__":
    train_vec_file = "data/train.vec"
    train_txt_file = "data/train.txt"
    dev_vec_file = "data/dev.vec"
    dev_txt_file = "data/dev.txt"

    all_ivectors, all_labels = combine_data(
        train_vec_file, train_txt_file, dev_vec_file, dev_txt_file
    )

    best_model = select_model(all_ivectors, all_labels, n_epochs=3, k=3, verbose=True)

    # Todo: Needs to be implemented.
    final_model = train_best_model()
