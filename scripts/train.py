import argparse
from sklearn.svm import SVC
from typing import List
from typing import Tuple

from dataset import combine_data
from dataset import load_ivectors
from dataset import load_labels
from logistic_regression import LogisticRegression
from logistic_regression import LogisticRegressionTrainer
from metrics import Metrics
from svm import SVMTrainer


MODEL_CHOICE = {
    "LogisticRegression": (LogisticRegression, LogisticRegressionTrainer),
    "SVM": (SVC, SVMTrainer)
}


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

    for model_type, trainer_type in MODEL_CHOICE.values():
        trainer = trainer_type(
            model_type,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            log_interval=log_interval,
        )
        trainer.cross_validation(all_ivectors, all_labels, k=k, verbose=True)
        metrics_all_models.append(trainer.cv_metrics)

    best_model = compare_metrics(metrics_all_models)

    return best_model


def compare_metrics(metrics_all_models: List[List[Tuple[str, Metrics]]]):
    # Todo: Put the metrics of all CVs from all models in here and
    pass


def train_single_model(
    model_name,
    train_ivectors,
    train_labels,
    test_ivectors,
    test_labels,
    # Todo: Check if these metrics should be different for different models.
    # Maybe use lists of batch sizes and learning rates (e.g.).
    n_epochs: int = 10,
    batch_size: int = 10,
    lr: int = 0.01,
    log_interval: int = 50,
    verbose: bool = False,
):
    model_type, trainer_type = MODEL_CHOICE[model_name]

    trainer = trainer_type(
        model_type,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        log_interval=log_interval,
    )

    trainer.train(model_name, train_ivectors, train_labels, test_ivectors, test_labels, verbose=True
        )


def train_final_model(
    train_ivectors,
    test_labels,
    # Todo: Check if these metrics should be different for different models.
    # Maybe use lists of batch sizes and learning rates (e.g.).
    n_epochs: int = 10,
    batch_size: int = 10,
    lr: int = 0.01,
    log_interval: int = 50,
    k: int = 10,
    verbose: bool = False,
):
    # TODO: In the train fuctions (e.g. in LogisticRegressionTrainer) I must
    # change the behaviour so that testing is optional.
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # argp.add_argument("config_file", help="A configuration file that specifies training parameters.")
    parser.add_argument(
        "-s",
        "--single_model",
        choices=["LogisticRegression"],
        help="Train a single model.",
    )
    parser.add_argument(
        "-f",
        "--final_model",
        choices=["LogisticRegression", "SVC"],
        help="Train a single model with the entire dataset",
    )

    # If cross validation is run, the original split cannot be used (hence
    # the mututally exclusive group).
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--cross_val", action="store_true", help="Select a model with cross_validation."
    )
    group.add_argument(
        "-o",
        "--original_split",
        action="store_true",
        help="Use the original train-test split given by VarDial.",
    )

    args = parser.parse_args()

    train_vec_file = "data/train.vec"
    train_txt_file = "data/train.txt"
    test_vec_file = "data/dev.vec"
    test_txt_file = "data/dev.txt"

    if args.original_split:

        train_ivectors = load_ivectors(train_vec_file)
        train_labels = load_labels(train_txt_file)
        test_ivectors = load_ivectors(test_vec_file)
        test_labels = load_labels(test_txt_file)

        if args.single_model:
            train_single_model(
                args.single_model,
                train_ivectors,
                train_labels,
                test_ivectors,
                test_labels,
            )
        if args.final_model:
            # TODO: Must be implemented.
            train_final_model()

    else:
        all_ivectors, all_labels = combine_data(
            train_vec_file, train_txt_file, test_vec_file, test_txt_file
        )

        if args.cross_val:
            best_model = select_model(
                # TODO: Make parameters configurable.
                all_ivectors, all_labels, n_epochs=3, k=3, verbose=True
            )

        if args.single_model:

            # TODO: Introduce train_test_split functionality (set parameter somewhere).
            train_single_model(
                args.single_model,
            )

        if args.final_model:
            # TODO: Must be implemented.
            train_final_model()
